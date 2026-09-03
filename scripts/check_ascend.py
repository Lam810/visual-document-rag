#!/usr/bin/env python
"""Diagnose Ascend NPU readiness, stage by stage.

Run this on the Ascend host before the app. It reports what each stage of the
pipeline will actually run on, so a silent CPU fallback shows up as a warning
here instead of as a slow job later.

    python scripts/check_ascend.py
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OK, WARN, FAIL, INFO = "  OK  ", " WARN ", " FAIL ", " INFO "
_results = []


def report(status: str, label: str, detail: str = "") -> None:
    _results.append((status, label))
    line = f"[{status}] {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


class DeviceOpenTimeout(Exception):
    pass


@contextmanager
def _time_limit(seconds: int):
    """Bound a blocking call with SIGALRM.

    Verified on real Ascend hardware (Ascend 910C, CANN 8.5.0, two different
    physical nodes) that a stuck HDC handshake makes the first real device
    touch - not ``is_available()``, which just queries the driver - hang
    indefinitely rather than raise. Both times it silently consumed an entire
    Slurm allocation (~13-15 min) with no output, because the hang happens
    inside a C++ extension that a plain Python try/except cannot interrupt
    once entered. A wall-clock alarm is what turns that into a diagnosable
    failure instead of a wasted allocation.
    """

    def _on_alarm(signum, frame):
        raise DeviceOpenTimeout(f"timed out after {seconds}s")

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def check_driver() -> None:
    section("Driver and CANN")
    npu_smi = shutil.which("npu-smi")
    if not npu_smi:
        report(FAIL, "npu-smi not on PATH", "Ascend driver is not installed or not sourced.")
        return
    try:
        out = subprocess.run(
            [npu_smi, "info"], capture_output=True, text=True, timeout=30
        )
        if out.returncode == 0:
            chips = [
                line.strip()
                for line in out.stdout.splitlines()
                if "910" in line or "310" in line
            ]
            report(OK, "npu-smi info", chips[0] if chips else "driver responding")
        else:
            report(FAIL, "npu-smi info failed", out.stderr.strip()[:200])
    except Exception as exc:
        report(FAIL, "npu-smi info errored", str(exc))

    cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if cann:
        report(OK, "CANN toolkit env", cann)
    else:
        report(
            WARN,
            "ASCEND_HOME_PATH unset",
            "source /usr/local/Ascend/ascend-toolkit/set_env.sh",
        )

    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    report(INFO, "ASCEND_RT_VISIBLE_DEVICES", visible or "(unset - all devices visible)")


def check_torch_npu() -> bool:
    section("PyTorch + torch_npu")
    try:
        import torch
    except ImportError:
        report(FAIL, "torch not installed")
        return False
    report(OK, "torch", torch.__version__)

    try:
        import torch_npu
    except ImportError:
        report(FAIL, "torch_npu not installed", "pip install torch-npu==<matching torch version>")
        return False

    npu_version = getattr(torch_npu, "__version__", "unknown")
    report(OK, "torch_npu", npu_version)

    base_torch = torch.__version__.split("+")[0]
    if not npu_version.startswith(base_torch.rsplit(".", 1)[0]):
        report(
            WARN,
            "torch / torch_npu version skew",
            f"torch {base_torch} vs torch_npu {npu_version}; these are expected to match",
        )

    try:
        available = torch.npu.is_available()
    except Exception as exc:
        report(FAIL, "torch.npu.is_available() raised", str(exc))
        return False

    if not available:
        report(FAIL, "torch.npu.is_available() is False", "Driver/CANN/torch_npu mismatch.")
        return False

    count = torch.npu.device_count()
    try:
        name = torch.npu.get_device_name(0)
    except Exception:
        name = "unknown"
    report(OK, "NPU visible", f"{count} device(s), device 0 = {name}")

    try:
        with _time_limit(60):
            a = torch.randn(256, 256).npu()
            result = (a @ a).sum().item()
        report(OK, "NPU matmul smoke test", f"sum={result:.4f}")
    except DeviceOpenTimeout as exc:
        report(
            FAIL,
            "NPU matmul timed out",
            f"{exc}. is_available() and get_device_name() both returned instantly, "
            "so the driver responds - this hangs specifically on opening the device "
            "for compute (HDC handshake). Not a torch_npu install problem. Try a "
            "different node, check whether the scheduler sets "
            "ASCEND_RT_VISIBLE_DEVICES for this job, and ask cluster support about "
            "per-job device isolation if it reproduces elsewhere.",
        )
        return False
    except Exception as exc:
        report(FAIL, "NPU matmul failed", str(exc))
        return False
    return True


def check_shim() -> bool:
    section("CUDA -> NPU shim")
    try:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError:
        report(
            WARN,
            "transfer_to_npu unavailable",
            "Formula recognition (UniMERNet) hardcodes torch.cuda and will run on CPU.",
        )
        return False
    import torch

    if torch.cuda.is_available():
        report(OK, "transfer_to_npu active", "torch.cuda calls now resolve to the NPU")
        return True
    report(WARN, "transfer_to_npu imported but torch.cuda still unavailable")
    return False


def check_paddle() -> None:
    section("PaddlePaddle (OCR stage)")
    try:
        import paddle
    except ImportError:
        report(WARN, "paddlepaddle not installed", "OCR stage will not run at all.")
        return
    report(OK, "paddle", paddle.__version__)
    try:
        customs = paddle.device.get_all_custom_device_type()
    except Exception as exc:
        report(WARN, "custom device probe failed", str(exc))
        return
    if "npu" in (customs or []):
        report(OK, "paddle-custom-npu", "OCR can run on the NPU")
        try:
            # Same HDC device-open hazard as the PyTorch smoke test above.
            with _time_limit(60):
                paddle.set_device("npu:0")
                tensor = paddle.randn([64, 64])
                mean = float((tensor @ tensor).mean())
            report(OK, "paddle NPU smoke test", f"mean={mean:.4f}")
        except DeviceOpenTimeout as exc:
            report(WARN, "paddle NPU compute timed out", f"{exc}. See the PyTorch timeout above.")
        except Exception as exc:
            report(WARN, "paddle NPU compute failed", str(exc))
    else:
        report(
            WARN,
            "paddle-custom-npu missing",
            "OCR falls back to CPU. Install paddle-custom-npu to move it to the NPU.",
        )


def check_pipeline_assets() -> None:
    section("Project assets")
    import config

    kit = config.PDF_EXTRACT_KIT_DIR
    if kit.exists():
        report(OK, "PDF-Extract-Kit", str(kit))
    else:
        report(FAIL, "PDF-Extract-Kit missing", f"Expected at {kit} (set VDR_PDF_EXTRACT_KIT).")
        return

    expected = [
        kit / "models" / "Layout" / "YOLO" / "doclayout_yolo_ft.pt",
        kit / "models" / "MFD" / "YOLO" / "yolo_v8_ft.pt",
        kit / "models" / "MFR" / "unimernet_tiny",
        kit / "models" / "OCR" / "PaddleOCR" / "det" / "ch_PP-OCRv4_det",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        report(WARN, "some model weights missing", "\n         ".join(missing))
    else:
        report(OK, "model weights present")


def check_resolution() -> None:
    section("Resolved configuration")
    import device as device_mod

    info = device_mod.bootstrap(force=True)
    report(INFO, "selected backend", info.describe())
    print()
    print("  Stage                  Runs on")
    print("  ---------------------  -------------------------")
    stages = [
        ("layout detection", info.third_party_device),
        ("formula detection", info.third_party_device),
        (
            "formula recognition",
            info.torch_device
            if info.backend != "npu" or info.cuda_shim_active
            else "cpu (shim inactive)",
        ),
        ("OCR (PaddleOCR)", info.paddle_device or "not installed"),
        ("embeddings", info.modelscope_device),
    ]
    for name, target in stages:
        print(f"  {name:<21}  {target}")


def main() -> int:
    print("Ascend readiness check for visual-document-rag")
    check_driver()
    npu_ok = check_torch_npu()
    if npu_ok:
        check_shim()
    check_paddle()
    check_pipeline_assets()
    check_resolution()

    section("Summary")
    failures = sum(1 for status, _ in _results if status == FAIL)
    warnings = sum(1 for status, _ in _results if status == WARN)
    print(f"{failures} failure(s), {warnings} warning(s)")

    # Compute readiness and asset readiness fail for different reasons and have
    # different fixes, so report them separately.
    if not npu_ok:
        print("\nNPU compute is not usable; every stage will run on CPU.")
    elif warnings:
        print("\nNPU compute works. Some stages still fall back to CPU - see warnings.")
    else:
        print("\nNPU compute works and every stage can use it.")
    if failures:
        print("Resolve the failures above before running the app.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
