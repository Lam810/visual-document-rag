"""Runtime configuration, resolved from environment variables.

Every path used to be hardcoded to a rented-GPU layout (``/root/autodl-tmp/...``),
which made the project unrunnable anywhere else. All of it now defaults to
locations inside the repository and can be redirected per deployment.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _path(env: str, default: Path) -> Path:
    raw = os.environ.get(env)
    return Path(raw).expanduser().resolve() if raw else default


def _flag(env: str, default: bool = False) -> bool:
    raw = os.environ.get(env)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(env: str, default: int) -> int:
    try:
        return int(os.environ.get(env, default))
    except ValueError:
        return default


# --- Data locations ---------------------------------------------------------
INPUT_DIR = _path("VDR_INPUT_DIR", BASE_DIR / "input")
OUTPUT_DIR = _path("VDR_OUTPUT_DIR", BASE_DIR / "output")
DATA_DIR = _path("VDR_DATA_DIR", BASE_DIR / "data")
CHROMA_DIR = _path("VDR_CHROMA_DIR", BASE_DIR / "chroma_db")
#: Where ModelScope caches downloaded weights.
MODEL_CACHE_DIR = _path("VDR_MODEL_CACHE", BASE_DIR / ".model_cache")

# --- Models -----------------------------------------------------------------
EMBEDDING_MODEL_ID = os.environ.get(
    "VDR_EMBEDDING_MODEL", "damo/nlp_corom_sentence-embedding_chinese-base"
)
EMBEDDING_MODEL_REVISION = os.environ.get("VDR_EMBEDDING_REVISION") or None
EMBEDDING_BATCH_SIZE = _int("VDR_EMBEDDING_BATCH_SIZE", 32)

# --- Document pipeline ------------------------------------------------------
#: Checkout of https://github.com/opendatalab/PDF-Extract-Kit.
PDF_EXTRACT_KIT_DIR = _path("VDR_PDF_EXTRACT_KIT", BASE_DIR / "PDF-Extract-Kit")
#: Base pipeline config; device settings are injected at runtime.
PIPELINE_CONFIG = _path("VDR_PIPELINE_CONFIG", BASE_DIR / "pdf2markdown.yaml")
#: Generated per-run copy of the above, with the resolved device filled in.
GENERATED_CONFIG = _path("VDR_GENERATED_CONFIG", BASE_DIR / ".pdf2markdown.runtime.yaml")

CHUNK_SIZE = _int("VDR_CHUNK_SIZE", 500)
CHUNK_OVERLAP = _int("VDR_CHUNK_OVERLAP", 50)

# --- Server -----------------------------------------------------------------
HOST = os.environ.get("VDR_HOST", "127.0.0.1")
PORT = _int("VDR_PORT", 6006)
DEBUG = _flag("VDR_DEBUG", False)
MAX_UPLOAD_MB = _int("VDR_MAX_UPLOAD_MB", 100)
#: Rough throughput estimate used only for the progress hint in the UI.
SECONDS_PER_MB = _int("VDR_SECONDS_PER_MB", 30)


def ensure_directories() -> None:
    for directory in (INPUT_DIR, OUTPUT_DIR, DATA_DIR, MODEL_CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def apply_model_cache_env() -> None:
    """Point ModelScope/HuggingFace caches at :data:`MODEL_CACHE_DIR`."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODELSCOPE_CACHE", str(MODEL_CACHE_DIR))
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "huggingface"))
