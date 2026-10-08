#!/usr/bin/env python3
"""Check an NPU elementwise add independently of a PaddleOCR model."""

import paddle


def main() -> None:
    devices = paddle.device.get_all_custom_device_type()
    if "npu" not in devices:
        raise RuntimeError(f"NPU custom device unavailable: {devices}")
    paddle.set_device("npu:0")

    lhs = paddle.ones([1, 32, 64, 64], dtype="float32")
    rhs = paddle.full([1, 32, 64, 64], 2.0, dtype="float32")
    result = paddle.add(lhs, rhs)
    actual = float(result.mean().numpy().item())
    print(f"device={paddle.get_device()} add_mean={actual}", flush=True)
    if actual != 3.0:
        raise RuntimeError(f"incorrect NPU add result: {actual}")
    print("NPU_ELEMENTWISE_ADD_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
