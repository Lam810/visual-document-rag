"""Driving PDF-Extract-Kit: config generation and process launch.

PDF-Extract-Kit exposes a ``device`` key per model, but only for some models,
and its formula-recognition stage ignores it entirely. This module centralises
what we *can* configure and hands the rest to the CUDA->NPU shim applied in
:mod:`device`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml

import config
import device as device_mod

logger = logging.getLogger(__name__)

#: Tasks that read ``device`` out of their ``model_config``.
#:
#: ``formula_recognition`` is listed too, but UniMERNet currently ignores it and
#: hardcodes ``torch.cuda``; it reaches the NPU via the shim in :mod:`device`,
#: and the key is set so it starts working if upstream ever honours it.
_DEVICE_AWARE_TASKS = ("layout_detection", "formula_detection", "formula_recognition")

#: PaddleOCR takes boolean per-backend flags instead of a device string.
_OCR_TASK = "ocr"


def _resolve_model_paths(node: Any, kit_dir: Path) -> Any:
    """Rewrite the ``PDF-Extract-Kit/...`` path prefixes to the real checkout.

    The committed config assumed the kit lived in a fixed subdirectory. Honour
    ``VDR_PDF_EXTRACT_KIT`` instead so the checkout can live anywhere.
    """
    if isinstance(node, dict):
        return {key: _resolve_model_paths(value, kit_dir) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_model_paths(item, kit_dir) for item in node]
    if isinstance(node, str) and node.startswith("PDF-Extract-Kit/"):
        return str(kit_dir / node[len("PDF-Extract-Kit/") :])
    return node


def _ocr_device_flags(device_info: device_mod.DeviceInfo) -> Dict[str, bool]:
    """Translate the resolved device into PaddleOCR's backend flags.

    PaddleOCR dispatches with ``if use_gpu: ... elif use_npu: ...``, so the two
    are not independent - leaving ``use_gpu`` true on Ascend wins the branch,
    then ``check_gpu`` quietly resets it because Paddle is not compiled with
    CUDA, and OCR ends up on the CPU with no warning. ``use_gpu`` must therefore
    be false for ``use_npu`` to take effect.

    PaddleOCR's ``ModifiedPaddleOCR`` forwards the whole ``model_config`` as
    kwargs, so these land directly on the PaddleOCR constructor.
    """
    paddle_device = device_info.paddle_device or "cpu"
    if paddle_device.startswith("npu"):
        return {"use_gpu": False, "use_npu": True}
    if paddle_device.startswith("gpu"):
        return {"use_gpu": True, "use_npu": False}
    return {"use_gpu": False, "use_npu": False}


def build_runtime_config(
    device_info: Optional[device_mod.DeviceInfo] = None,
    destination: Optional[Path] = None,
    inputs: Optional[Path] = None,
) -> Path:
    """Materialise a pipeline config with the resolved device filled in.

    ``inputs`` overrides the directory the pipeline reads from, which is how a
    single-document run is scoped.
    """
    device_info = device_info or device_mod.get_device()
    destination = destination or config.GENERATED_CONFIG

    with open(config.PIPELINE_CONFIG, "r", encoding="utf-8") as handle:
        pipeline_config: Dict[str, Any] = yaml.safe_load(handle) or {}

    pipeline_config = _resolve_model_paths(pipeline_config, config.PDF_EXTRACT_KIT_DIR)
    pipeline_config["inputs"] = str(inputs or config.INPUT_DIR)
    pipeline_config["outputs"] = str(config.OUTPUT_DIR)

    tasks = pipeline_config.get("tasks") or {}
    for task_name, task in tasks.items():
        model_config = task.setdefault("model_config", {})
        if task_name in _DEVICE_AWARE_TASKS:
            model_config["device"] = device_info.third_party_device
        if task_name == _OCR_TASK:
            model_config.update(_ocr_device_flags(device_info))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        yaml.safe_dump(pipeline_config, handle, allow_unicode=True, sort_keys=False)

    logger.info(
        "Pipeline config written to %s (device=%s)",
        destination,
        device_info.third_party_device,
    )
    return destination


def kit_entrypoint() -> Path:
    return config.PDF_EXTRACT_KIT_DIR / "project" / "pdf2markdown" / "scripts" / "run_project.py"


@contextmanager
def _scoped_inputs(document: Optional[Path]) -> Iterator[Path]:
    """Yield an input directory containing only ``document``.

    PDF-Extract-Kit reads a whole directory, so pointing it at the upload folder
    re-runs the entire vision pipeline over every previously uploaded PDF on
    each new upload - O(n) full extractions per document. Staging the one file
    in a temporary directory keeps each upload independent.
    """
    if document is None:
        yield config.INPUT_DIR
        return
    staging = Path(tempfile.mkdtemp(prefix="vdr-input-"))
    try:
        try:
            # Same filesystem in the default layout, so this is usually free.
            (staging / document.name).hardlink_to(document)
        except (OSError, AttributeError):
            shutil.copy2(document, staging / document.name)
        yield staging
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run(
    device_info: Optional[device_mod.DeviceInfo] = None,
    document: Optional[Path] = None,
    timeout: Optional[int] = None,
):
    """Run the extraction pipeline in a subprocess and return the result.

    The subprocess is launched through :mod:`scripts.run_pdf2markdown` so the
    CUDA->NPU shim is installed before PDF-Extract-Kit imports PyTorch - the
    shim only affects modules imported after it.

    Passing ``document`` restricts the run to that single PDF; omitting it
    processes everything in the input directory.
    """
    device_info = device_info or device_mod.get_device()
    entrypoint = kit_entrypoint()
    if not entrypoint.exists():
        raise FileNotFoundError(
            f"PDF-Extract-Kit not found at {config.PDF_EXTRACT_KIT_DIR}. "
            "Clone it there or set VDR_PDF_EXTRACT_KIT."
        )

    launcher = config.BASE_DIR / "scripts" / "run_pdf2markdown.py"
    with _scoped_inputs(document) as inputs:
        runtime_config = build_runtime_config(device_info, inputs=inputs)
        return subprocess.run(
            [sys.executable, str(launcher), "--config", str(runtime_config)],
            check=True,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
