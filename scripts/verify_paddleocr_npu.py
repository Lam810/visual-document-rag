#!/usr/bin/env python3
"""Run a real PP-OCRv4 detection and recognition pass on an Ascend NPU.

Run this inside an allocated NPU job after sourcing the CANN toolkit and NNAL
ATB environments. The model directories must contain inference.pdmodel and
inference.pdiparams. This deliberately checks more than device selection: it
performs a Paddle tensor operation and OCR inference, then checks recognized
text from a known image.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--det-model-dir", required=True, type=Path)
    parser.add_argument("--rec-model-dir", required=True, type=Path)
    parser.add_argument("--cls-model-dir", required=True, type=Path)
    parser.add_argument("--expect-text", default="春节")
    args = parser.parse_args()

    for path in (args.image, args.det_model_dir, args.rec_model_dir, args.cls_model_dir):
        if not path.exists():
            parser.error(f"missing input: {path}")

    for package in ("paddlepaddle", "paddle-custom-npu", "paddleocr"):
        print(f"{package}={version(package)}", flush=True)

    import paddle

    custom_devices = paddle.device.get_all_custom_device_type()
    print(f"custom_devices={custom_devices}", flush=True)
    if "npu" not in custom_devices:
        raise RuntimeError("Paddle did not detect the NPU custom device")

    paddle.set_device("npu:0")
    tensor = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    total = float(paddle.matmul(tensor, tensor).sum().numpy().item())
    print(f"paddle_device={paddle.get_device()} npu_matmul_sum={total}", flush=True)
    if total != 54.0:
        raise RuntimeError(f"unexpected NPU matmul result: {total}")

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        use_gpu=False,
        use_npu=True,
        use_angle_cls=False,
        det_model_dir=str(args.det_model_dir),
        rec_model_dir=str(args.rec_model_dir),
        cls_model_dir=str(args.cls_model_dir),
        lang="ch",
        show_log=False,
    )
    result = ocr.ocr(str(args.image), cls=False)
    lines = [
        {"text": line[1][0], "score": float(line[1][1])}
        for page in result or []
        for line in page or []
    ]
    print(f"ocr_device={paddle.get_device()} ocr_lines={len(lines)}", flush=True)
    print(json.dumps(lines[:20], ensure_ascii=False), flush=True)
    if not lines:
        raise RuntimeError("PaddleOCR returned no text")
    if args.expect_text and not any(args.expect_text in line["text"] for line in lines):
        raise RuntimeError(f"expected text {args.expect_text!r} was not recognized")
    print("PADDLEOCR_NPU_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
