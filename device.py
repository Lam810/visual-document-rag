"""Backend detection and device selection.

The pipeline in this project runs three independent stacks:

* **PyTorch** - layout detection (YOLO), formula detection (YOLO), formula
  recognition (UniMERNet), and the sentence-embedding model.
* **ModelScope** - wraps PyTorch, but validates device strings against its own
  whitelist.
* **PaddlePaddle** - PaddleOCR, which does not go through PyTorch at all.

Each of them names accelerators differently, and two of them refuse to accept
``npu`` as a device string. This module hides those differences behind a single
:func:`bootstrap` call so the rest of the codebase never branches on hardware.

Supported backends: Ascend NPU (via ``torch_npu``), CUDA, Apple MPS, CPU.

Environment variables
---------------------
``VDR_DEVICE``            ``auto`` (default), ``npu``, ``cuda``, ``mps``, ``cpu``.
``VDR_DEVICE_ID``         Ordinal of the accelerator to use (default ``0``).
``VDR_DISABLE_CUDA_SHIM`` Set to ``1`` to skip the CUDA->NPU compatibility shim.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_BOOTSTRAPPED: Optional["DeviceInfo"] = None

#: Backends we know how to target, in the order we prefer them.
_PREFERENCE = ("npu", "cuda", "mps", "cpu")


@dataclass
class DeviceInfo:
    """Resolved accelerator, expressed in each framework's own dialect."""

    backend: str = "cpu"
    index: int = 0
    #: Device string for ``torch.device(...)`` - e.g. ``npu:0``, ``cuda:0``.
    torch_device: str = "cpu"
    #: Device string ModelScope's ``pipeline(device=...)`` will accept.
    modelscope_device: str = "cpu"
    #: Device string for ``paddle.set_device(...)``, or ``None`` if unavailable.
    paddle_device: Optional[str] = None
    #: True when the CUDA->NPU shim is active, so ``torch.cuda`` means NPU.
    cuda_shim_active: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_accelerated(self) -> bool:
        return self.backend != "cpu"

    @property
    def third_party_device(self) -> str:
        """Device string to hand to dependencies that only parse CUDA names.

        Ultralytics/doclayout-YOLO route their ``device=`` argument through
        ``select_device``, which understands ``cpu``, ``mps`` and CUDA ordinals
        and rejects ``npu:0``. When the CUDA->NPU shim is active the honest
        answer is still ``cuda:N``, because every CUDA call it makes is being
        redirected to the NPU anyway.
        """
        if self.backend == "npu" and self.cuda_shim_active:
            return f"cuda:{self.index}"
        return self.torch_device

    def describe(self) -> str:
        bits = [f"backend={self.backend}", f"torch={self.torch_device}"]
        if self.backend == "npu":
            bits.append(f"cuda_shim={'on' if self.cuda_shim_active else 'off'}")
        if self.paddle_device:
            bits.append(f"paddle={self.paddle_device}")
        name = self.details.get("device_name")
        if name:
            bits.append(f"name={name}")
        return " ".join(bits)


def _requested_backend() -> str:
    value = os.environ.get("VDR_DEVICE", "auto").strip().lower()
    if value in ("", "auto"):
        return "auto"
    if value == "gpu":  # common shorthand; resolve it like "auto" would
        return "auto"
    if value not in _PREFERENCE:
        logger.warning("Unknown VDR_DEVICE=%r, falling back to auto-detection", value)
        return "auto"
    return value


def _device_id() -> int:
    try:
        return int(os.environ.get("VDR_DEVICE_ID", "0"))
    except ValueError:
        logger.warning("VDR_DEVICE_ID is not an integer, using 0")
        return 0


def _try_import_torch_npu():
    """Import ``torch_npu`` and report whether a usable NPU is present.

    ``import torch_npu`` is what registers the ``npu`` device with PyTorch, so it
    has to happen before any ``torch.npu`` attribute access.

    Catching only ``ImportError`` is not enough here. Since PyTorch 2.6, ``import
    torch`` auto-loads ``torch_npu`` through its device-backend entry point, so a
    torch_npu that is installed but broken - a missing transitive dependency, a
    CANN version mismatch - makes ``import torch`` itself raise ``RuntimeError``.
    Letting that escape would take down a process that could have run on CPU.
    """
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
    except ImportError:
        return None, None
    except Exception as exc:
        logger.warning(
            "torch_npu is installed but failed to load (%s). Falling back to CPU. "
            "Set TORCH_DEVICE_BACKEND_AUTOLOAD=0 to stop PyTorch importing it.",
            exc,
        )
        return None, None
    try:
        import torch

        if torch.npu.is_available():
            return torch, torch_npu
    except Exception as exc:  # driver present but unusable, CANN mismatch, ...
        logger.warning("torch_npu imported but no usable NPU: %s", exc)
    return None, None


def _apply_cuda_shim() -> bool:
    """Route ``torch.cuda.*`` calls to the NPU.

    Two dependencies make this necessary rather than cosmetic:

    * PDF-Extract-Kit's UniMERNet model hardcodes
      ``torch.device("cuda" if torch.cuda.is_available() else "cpu")`` with no
      config override, so without the shim formula recognition silently runs on
      CPU.
    * ModelScope's ``verify_device`` asserts the device string starts with
      ``cpu``/``cuda``/``gpu``, so ``npu:0`` raises before it reaches PyTorch.
      With the shim, asking ModelScope for ``gpu:0`` lands on the NPU.

    Must run before the libraries that touch ``torch.cuda`` are imported.
    """
    if os.environ.get("VDR_DISABLE_CUDA_SHIM", "").strip() == "1":
        logger.info("CUDA->NPU shim disabled by VDR_DISABLE_CUDA_SHIM=1")
        return False
    try:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError:
        logger.warning(
            "torch_npu.contrib.transfer_to_npu is unavailable; components that "
            "hardcode torch.cuda (UniMERNet formula recognition) will stay on CPU"
        )
        return False

    # Verify it actually took effect rather than trusting the import. The device
    # strings we hand to ModelScope and Ultralytics are only correct while the
    # shim is live, so a silent no-op here would send both to a real CUDA device
    # that does not exist.
    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning(
                "transfer_to_npu was imported but torch.cuda is still "
                "unavailable; treating the shim as inactive"
            )
            return False
    except Exception as exc:
        logger.warning("Could not verify the CUDA->NPU shim: %s", exc)
        return False
    return True


def _paddle_device(backend: str, index: int) -> Optional[str]:
    """Best available PaddlePaddle device string, or ``None`` if Paddle is absent.

    PaddleOCR is the one stage that never touches PyTorch, so Ascend support here
    depends on ``paddle-custom-npu`` rather than ``torch_npu``. When it is not
    installed we return ``cpu`` so OCR still runs, just slower.
    """
    try:
        import paddle
    except ImportError:
        return None
    if backend == "npu":
        try:
            if "npu" in paddle.device.get_all_custom_device_type():
                return f"npu:{index}"
        except Exception as exc:
            logger.debug("Paddle custom device probe failed: %s", exc)
        logger.warning(
            "paddle-custom-npu not detected; PaddleOCR will run on CPU while the "
            "PyTorch stages use the NPU"
        )
        return "cpu"
    if backend == "cuda":
        try:
            if paddle.device.is_compiled_with_cuda():
                return f"gpu:{index}"
        except Exception:
            pass
    return "cpu"


def _npu_details(torch_mod, index: int) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    try:
        details["device_name"] = torch_mod.npu.get_device_name(index)
    except Exception:
        pass
    try:
        details["device_count"] = torch_mod.npu.device_count()
    except Exception:
        pass
    try:
        import torch_npu

        details["torch_npu_version"] = getattr(torch_npu, "__version__", "unknown")
    except Exception:
        pass
    details["cann_home"] = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME"
    )
    return details


def bootstrap(force: bool = False) -> DeviceInfo:
    """Detect the accelerator and prepare the process to use it.

    Idempotent: later calls return the cached result unless ``force`` is set.
    Call this as early as possible - the CUDA->NPU shim only affects modules
    imported after it runs.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED is not None and not force:
        return _BOOTSTRAPPED

    requested = _requested_backend()
    index = _device_id()
    info = DeviceInfo(index=index)

    if requested == "cpu":
        info.details["reason"] = "VDR_DEVICE=cpu"
        _BOOTSTRAPPED = info
        return info

    if requested in ("auto", "npu"):
        torch_mod, _ = _try_import_torch_npu()
        if torch_mod is not None:
            # ASCEND_RT_VISIBLE_DEVICES already remapped the ordinals if it was
            # set, so index is relative to the visible set, same as CUDA.
            info.backend = "npu"
            info.torch_device = f"npu:{index}"
            # Only valid because the shim aliases cuda->npu; see _apply_cuda_shim.
            info.cuda_shim_active = _apply_cuda_shim()
            info.modelscope_device = f"gpu:{index}" if info.cuda_shim_active else "cpu"
            info.details.update(_npu_details(torch_mod, index))
            info.paddle_device = _paddle_device("npu", index)
            _BOOTSTRAPPED = info
            logger.info("Selected device: %s", info.describe())
            return info
        if requested == "npu":
            logger.error(
                "VDR_DEVICE=npu but no usable Ascend NPU was found; falling back "
                "to CPU. Run scripts/check_ascend.py to diagnose."
            )
            info.details["reason"] = "npu requested but unavailable"
            _BOOTSTRAPPED = info
            return info

    if requested in ("auto", "cuda"):
        try:
            import torch

            # Same backend-autoload hazard as above; a broken accelerator plugin
            # must not prevent the CPU path from being selected.
            if torch.cuda.is_available():
                info.backend = "cuda"
                info.torch_device = f"cuda:{index}"
                info.modelscope_device = f"gpu:{index}"
                info.paddle_device = _paddle_device("cuda", index)
                try:
                    info.details["device_name"] = torch.cuda.get_device_name(index)
                    info.details["device_count"] = torch.cuda.device_count()
                except Exception:
                    pass
                _BOOTSTRAPPED = info
                logger.info("Selected device: %s", info.describe())
                return info
        except ImportError:
            logger.warning("PyTorch is not installed; running on CPU")
        except Exception as exc:
            logger.warning("PyTorch failed to load (%s); running on CPU", exc)

    if requested in ("auto", "mps"):
        try:
            import torch

            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                info.backend = "mps"
                info.torch_device = "mps"
                info.modelscope_device = "cpu"  # ModelScope has no MPS device string
                info.paddle_device = _paddle_device("mps", index)
                _BOOTSTRAPPED = info
                logger.info("Selected device: %s", info.describe())
                return info
        except ImportError:
            pass

    info.paddle_device = _paddle_device("cpu", index)
    _BOOTSTRAPPED = info
    logger.info("Selected device: %s", info.describe())
    return info


def get_device() -> DeviceInfo:
    """Return the resolved device, bootstrapping on first use."""
    return bootstrap()


def apply_paddle_device(info: Optional[DeviceInfo] = None) -> Optional[str]:
    """Point PaddlePaddle at the resolved device. Returns the device actually set."""
    info = info or get_device()
    if not info.paddle_device:
        return None
    try:
        import paddle

        paddle.set_device(info.paddle_device)
        return info.paddle_device
    except Exception as exc:
        logger.warning("Could not set Paddle device %s: %s", info.paddle_device, exc)
        return None
