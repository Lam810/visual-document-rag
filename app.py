"""Flask front-end: upload a PDF, watch it become Markdown, ask questions of it."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import config
import device as device_mod
import pipeline
from rag import RAGSystem

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

config.ensure_directories()
config.apply_model_cache_env()
DEVICE = device_mod.bootstrap()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

rag_system = RAGSystem()


def _markdown_names() -> list[str]:
    return sorted(p.name for p in config.DATA_DIR.glob("*.md"))


def _resolve_in_data_dir(filename: str) -> Path | None:
    """Resolve ``filename`` inside the data directory, or ``None`` if it escapes.

    Rejects traversal (``../``), absolute paths and symlinks pointing outside,
    all of which the previous ``os.path.join`` version happily followed.
    """
    candidate = (config.DATA_DIR / filename).resolve()
    try:
        candidate.relative_to(config.DATA_DIR.resolve())
    except ValueError:
        return None
    if candidate.suffix != ".md" or not candidate.is_file():
        return None
    return candidate


@app.route("/")
def index():
    return render_template("index.html", documents=_markdown_names())


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "No selected file"}), 400
    if not uploaded.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    # secure_filename strips directory components, so an upload named
    # "../../etc/passwd.pdf" cannot write outside the input directory.
    safe_name = secure_filename(uploaded.filename)
    if not safe_name.lower().endswith(".pdf"):
        return jsonify({"error": "Invalid file name"}), 400

    # Size from the stream position rather than reading the file into memory.
    uploaded.stream.seek(0, 2)
    file_size = uploaded.stream.tell()
    uploaded.stream.seek(0)
    estimated_time = (file_size / (1024 * 1024)) * config.SECONDS_PER_MB

    input_path = config.INPUT_DIR / safe_name
    uploaded.save(input_path)

    stem = Path(safe_name).stem
    try:
        pipeline.run(DEVICE, document=input_path)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        logger.exception("PDF extraction failed")
        stderr = getattr(exc, "stderr", None)
        return jsonify({"error": f"PDF extraction failed: {stderr or exc}"}), 500

    produced = config.OUTPUT_DIR / f"{stem}.md"
    if not produced.exists():
        return jsonify({"error": "Extraction produced no Markdown output"}), 500

    destination = config.DATA_DIR / produced.name
    shutil.move(str(produced), str(destination))

    global rag_system
    rag_system = RAGSystem()
    rag_system.process_documents()

    return jsonify(
        {
            "message": "File processed successfully",
            "documents": _markdown_names(),
            "estimated_time": estimated_time,
            "markdown_content": destination.read_text(encoding="utf-8"),
        }
    )


@app.route("/documents", methods=["GET"])
def get_documents():
    return jsonify({"documents": _markdown_names()})


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400
    try:
        results = rag_system.search(data["question"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    except Exception as exc:
        logger.exception("Query failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify({"results": [{"content": doc.page_content} for doc in results]})


@app.route("/markdown/<path:filename>")
def get_markdown(filename):
    resolved = _resolve_in_data_dir(filename)
    if resolved is None:
        return jsonify({"error": "File not found"}), 404
    return jsonify({"content": resolved.read_text(encoding="utf-8")})


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "device": DEVICE.describe(),
            "backend": DEVICE.backend,
            "documents": len(_markdown_names()),
        }
    )


def main() -> None:
    print(f"Device: {DEVICE.describe()}")
    rag_system.process_documents()
    # Binds to localhost by default; set VDR_HOST=0.0.0.0 to expose it, and note
    # that VDR_DEBUG=1 enables the Werkzeug debugger, which executes code from
    # any client that can reach the port.
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)


if __name__ == "__main__":
    main()
