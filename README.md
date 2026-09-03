<div align="center">

# Visual Document RAG

**RAG that keeps reading a PDF's layout, not just its words.**
Formulas, tables, and figures survive the pipeline as themselves — not as flattened text.

[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Ascend NPU](https://img.shields.io/badge/Ascend%20NPU-910C%20tested-0f6f7d.svg)](ASCEND_VERIFICATION.md)

</div>

---

A retrieval-augmented question-answering system for **visually rich documents** — PDFs
where meaning lives in the layout, formulas, tables and figures rather than in a linear
stream of text.

Plain text extraction throws that structure away. This project runs a document through a
multi-stage vision pipeline first — layout detection, formula detection, formula
recognition, OCR — reassembles the result as structured Markdown, and only then chunks
and embeds it for retrieval. Questions are answered against a representation that still
knows a formula was a formula.

The pipeline is designed to run on **Huawei Ascend NPUs** as well as CUDA GPUs and CPU,
with per-stage device resolution rather than a single global flag. See
[Running on Ascend NPU](#running-on-ascend-npu).

---

## How it works

```
PDF ──► layout detection ──► formula detection ──► formula recognition ──► OCR
        (DocLayout-YOLO)     (YOLOv8)              (UniMERNet → LaTeX)     (PaddleOCR)
                                       │
                                       ▼
                          structured Markdown  ──►  chunk  ──►  embed  ──►  Chroma
                                                                                │
                                       question ──────────────────────────► retrieve
```

The vision stages come from [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit).
This project supplies the orchestration, the device abstraction, the retrieval layer and
the web front-end.

| Module | Responsibility |
| --- | --- |
| [`device.py`](device.py) | Detects the accelerator and reconciles four different device dialects |
| [`config.py`](config.py) | All paths and tunables, resolved from environment variables |
| [`pipeline.py`](pipeline.py) | Generates the per-run PDF-Extract-Kit config and launches it |
| [`rag.py`](rag.py) | Chunking, device-pinned embeddings, Chroma vector store |
| [`app.py`](app.py) | Flask upload / preview / query API |
| [`scripts/check_ascend.py`](scripts/check_ascend.py) | Per-stage NPU readiness diagnostic |

---

## Quick start

```bash
git clone https://github.com/Lam810/visual-document-rag.git
cd visual-document-rag

# 1. Vision models
git clone https://github.com/opendatalab/PDF-Extract-Kit.git
git lfs install
git clone https://www.modelscope.cn/opendatalab/pdf-extract-kit-1.0.git
mv ./pdf-extract-kit-1.0/models ./PDF-Extract-Kit

# 2. Environment
conda create -n visual-document-rag python=3.10
conda activate visual-document-rag
pip install -r requirements.txt

# 3. Accelerator (pick one)
pip install -r requirements-npu.txt              # Ascend NPU
pip install torch --index-url https://download.pytorch.org/whl/cu121   # NVIDIA

# 4. Run
python app.py                                    # http://127.0.0.1:6006
```

Upload a PDF, wait for extraction, then ask questions about it.

---

## Running on Ascend NPU

### Why this needs more than a device flag

The pipeline spans three frameworks that disagree about how to name an accelerator, and
two of the stages cannot be pointed at an NPU through configuration at all:

| Stage | Framework | How it picks a device | Ascend obstacle |
| --- | --- | --- | --- |
| Layout detection | Ultralytics/PyTorch | `device` in config | `select_device` parses CUDA ordinals; rejects `npu:0` |
| Formula detection | Ultralytics/PyTorch | `device` in config | same |
| Formula recognition | PyTorch | **hardcoded** `torch.device("cuda" if torch.cuda.is_available() else "cpu")` | no override exists |
| OCR | PaddlePaddle | `use_gpu` / `use_npu` flags | needs `paddle-custom-npu`; `use_gpu` wins the branch if left true |
| Embeddings | ModelScope → PyTorch | `pipeline(device=...)` | `verify_device` asserts the string starts with `cpu`/`cuda`/`gpu` |

Two of those — formula recognition and embeddings — will *silently* run on CPU rather
than raise, which is the failure mode this project is built to avoid.

The resolution is Ascend's official `torch_npu.contrib.transfer_to_npu` shim, which
rewrites `torch.cuda.*` calls to `torch.npu.*` at import time. With it active, the
hardcoded CUDA path lands on the NPU and ModelScope's `gpu:0` does too.
[`device.py`](device.py) applies the shim, **verifies it took effect**, and then emits
the correct device string per framework — falling back to honest `npu:` strings if the
shim is unavailable, rather than handing out CUDA names that would point nowhere.

Because the shim only affects modules imported after it,
[`scripts/run_pdf2markdown.py`](scripts/run_pdf2markdown.py) bootstraps the device before
delegating to PDF-Extract-Kit's entrypoint.

### Install

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# torch must be the CPU build: Ascend runs PyTorch through torch_npu, not CUDA.
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-npu.txt

# Optional, moves OCR onto the NPU too. Without it OCR runs on CPU.
pip install paddlepaddle==3.0.0
pip install paddle-custom-npu -i https://www.paddlepaddle.org.cn/packages/stable/npu/
```

`torch-npu`'s version tracks `torch`'s exactly, and each release targets a specific CANN
version — check the [matrix](https://gitee.com/ascend/pytorch#installation) before
changing either.

### Verify before running

```bash
python scripts/check_ascend.py
```

This reports each stage independently, so a CPU fallback shows up as a warning here
rather than as an unexplained slowdown later:

```
  Stage                  Runs on
  ---------------------  -------------------------
  layout detection       cuda:0        (shim → NPU)
  formula detection      cuda:0        (shim → NPU)
  formula recognition    npu:0
  OCR (PaddleOCR)        npu:0
  embeddings             gpu:0         (shim → NPU)
```

### Status

Tested on a real Ascend 910B cluster (CANN 8.5.0). Driver detection, `torch`/`torch_npu`
installation, and device metadata queries (`is_available`, `device_count`,
`get_device_name`) all work correctly. **Actual NPU compute is currently blocked by a
cluster-side issue**: opening the device for the first real operation
(`torch.npu.set_device`, or any tensor's `.npu()`) hangs indefinitely, reproduced
identically across two nodes and both `srun` and `sbatch`. This looks like a device
isolation gap in that cluster's Slurm GRES plugin, not a bug in this project — full
diagnostic timeline, what was ruled out, and what's still unverified as a result:
see [`ASCEND_VERIFICATION.md`](ASCEND_VERIFICATION.md).

The device-selection *logic* (which device string each framework receives, under every
backend/shim-state combination) was separately verified against stubbed `torch_npu` /
`paddle` modules and is correct. What's unverified is inference actually running, which
needs the cluster-side hang above resolved first.

Known gaps to expect once compute is unblocked:

- Ultralytics/DocLayout-YOLO device parsing is the least certain stage; if it rejects
  `cuda:0` under the shim, set `VDR_DISABLE_CUDA_SHIM=1` and compare.
- UniMERNet's `batch_size: 128` default is tuned for large-VRAM GPUs and will likely
  need lowering.
- Operators unsupported by CANN fall back to CPU per-op rather than failing loudly.

---

## Configuration

Everything is environment-driven; no paths are baked into the source.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VDR_DEVICE` | `auto` | `auto`, `npu`, `cuda`, `mps`, `cpu` |
| `VDR_DEVICE_ID` | `0` | Accelerator ordinal |
| `VDR_DISABLE_CUDA_SHIM` | unset | `1` disables the CUDA→NPU shim |
| `VDR_PDF_EXTRACT_KIT` | `./PDF-Extract-Kit` | PDF-Extract-Kit checkout |
| `VDR_DATA_DIR` | `./data` | Extracted Markdown |
| `VDR_INPUT_DIR` | `./input` | Uploaded PDFs |
| `VDR_OUTPUT_DIR` | `./output` | Pipeline scratch output |
| `VDR_CHROMA_DIR` | `./chroma_db` | Vector store |
| `VDR_MODEL_CACHE` | `./.model_cache` | ModelScope / HuggingFace cache |
| `VDR_EMBEDDING_MODEL` | `damo/nlp_corom_sentence-embedding_chinese-base` | Embedding model |
| `VDR_CHUNK_SIZE` / `VDR_CHUNK_OVERLAP` | `500` / `50` | Chunking |
| `VDR_HOST` / `VDR_PORT` | `127.0.0.1` / `6006` | Bind address |
| `VDR_DEBUG` | `0` | Werkzeug debugger — see below |
| `VDR_MAX_UPLOAD_MB` | `100` | Upload size limit |

On Ascend, `ASCEND_RT_VISIBLE_DEVICES` restricts which NPUs are visible; `VDR_DEVICE_ID`
indexes into that visible set, the same way it works for CUDA.

### Deployment note

The server binds to `127.0.0.1` and runs with the debugger off. Setting `VDR_HOST=0.0.0.0`
together with `VDR_DEBUG=1` exposes the Werkzeug debugger, which executes arbitrary code
for anyone who can reach the port — do not combine them on a shared machine. There is no
authentication layer; this is a research prototype, not a multi-tenant service.

---

## Acknowledgements

- [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) — layout, formula and OCR models
- [UniMERNet](https://github.com/opendatalab/UniMERNet) — formula recognition
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — text recognition
- [ModelScope](https://github.com/modelscope/modelscope) — embedding models
- [Ascend PyTorch adapter](https://gitee.com/ascend/pytorch) — `torch_npu`

Not affiliated with or endorsed by Huawei. "Ascend" and "CANN" are Huawei trademarks,
used here only to describe hardware compatibility.

## License

MIT
