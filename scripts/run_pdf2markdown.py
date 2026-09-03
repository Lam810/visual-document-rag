#!/usr/bin/env python
"""Launch PDF-Extract-Kit with the accelerator prepared first.

PDF-Extract-Kit's own entrypoint imports PyTorch immediately, and its
UniMERNet stage hardcodes ``torch.device("cuda" if torch.cuda.is_available()
else "cpu")``. On Ascend that resolves to CPU unless ``transfer_to_npu`` has
already patched ``torch.cuda``. Importing :mod:`device` before delegating is
what makes the NPU visible to that code.
"""

from __future__ import annotations

import argparse
import logging
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
import device as device_mod  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pipeline config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.apply_model_cache_env()

    info = device_mod.bootstrap()
    print(f"[run_pdf2markdown] device: {info.describe()}", flush=True)

    # PaddleOCR does not consult the PyTorch device at all.
    applied = device_mod.apply_paddle_device(info)
    if applied:
        print(f"[run_pdf2markdown] paddle device: {applied}", flush=True)

    entrypoint = (
        config.PDF_EXTRACT_KIT_DIR / "project" / "pdf2markdown" / "scripts" / "run_project.py"
    )
    if not entrypoint.exists():
        print(f"error: PDF-Extract-Kit entrypoint missing at {entrypoint}", file=sys.stderr)
        return 1

    # run_project.py resolves its imports relative to the kit checkout.
    sys.path.insert(0, str(config.PDF_EXTRACT_KIT_DIR))
    sys.argv = [str(entrypoint), "--config", args.config]
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
