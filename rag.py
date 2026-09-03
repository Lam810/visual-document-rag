"""Retrieval layer: chunk the extracted Markdown, embed it, and search it."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import config
import device as device_mod

# Caches must be redirected and the accelerator resolved before ModelScope or
# PyTorch are imported anywhere, so do both at module import time.
config.apply_model_cache_env()
_DEVICE = device_mod.bootstrap()

from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_text_splitters import MarkdownTextSplitter  # noqa: E402

logger = logging.getLogger(__name__)


class ModelScopeEmbeddings(Embeddings):
    """ModelScope sentence embeddings, pinned to a chosen device.

    ``langchain_community.embeddings.ModelScopeEmbeddings`` builds its pipeline
    without a ``device`` argument, so it always lands on CUDA-or-CPU and can
    never reach an Ascend NPU. This reimplementation passes the device through
    and, if ModelScope rejects it, relocates the underlying module by hand.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_revision: Optional[str] = None,
        batch_size: Optional[int] = None,
        device_info: Optional[device_mod.DeviceInfo] = None,
    ) -> None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.model_id = model_id or config.EMBEDDING_MODEL_ID
        self.model_revision = model_revision or config.EMBEDDING_MODEL_REVISION
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        self.device_info = device_info or device_mod.get_device()

        kwargs: Dict[str, Any] = {
            "task": Tasks.sentence_embedding,
            "model": self.model_id,
        }
        if self.model_revision:
            kwargs["model_revision"] = self.model_revision

        target = self.device_info.modelscope_device
        try:
            self.pipeline = pipeline(device=target, **kwargs)
            self.placement = target
        except Exception as exc:
            # ModelScope's verify_device only whitelists cpu/cuda/gpu, and older
            # releases reject anything else outright. Build on CPU and move the
            # module ourselves.
            logger.warning(
                "ModelScope rejected device=%r (%s); building on CPU and "
                "relocating manually",
                target,
                exc,
            )
            self.pipeline = pipeline(device="cpu", **kwargs)
            self.placement = self._relocate(self.device_info.torch_device)

        logger.info("Embedding model %s ready on %s", self.model_id, self.placement)

    def _relocate(self, torch_device: str) -> str:
        """Best-effort move of the pipeline's module onto ``torch_device``."""
        if torch_device == "cpu":
            return "cpu"
        try:
            import torch

            module = getattr(self.pipeline, "model", None)
            if module is None or not hasattr(module, "to"):
                logger.warning("Pipeline exposes no movable module; staying on CPU")
                return "cpu"
            module.to(torch.device(torch_device))
            # The pipeline moves its inputs to whatever `.device` says.
            if hasattr(self.pipeline, "device"):
                self.pipeline.device = torch.device(torch_device)
            return torch_device
        except Exception as exc:
            logger.warning("Could not relocate embedding model to %s: %s", torch_device, exc)
            return "cpu"

    def _embed(self, texts: List[str]) -> List[List[float]]:
        cleaned = [text.replace("\n", " ") for text in texts]
        vectors: List[List[float]] = []
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            result = self.pipeline(input={"source_sentence": batch})
            embeddings = result["text_embedding"]
            vectors.extend(
                embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)
            )
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class RAGSystem:
    """Chunk, embed and search the Markdown produced by the PDF pipeline."""

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self.data_dir = str(data_dir or config.DATA_DIR)
        self.device_info = _DEVICE
        self.embeddings = ModelScopeEmbeddings(device_info=self.device_info)
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
        self.db = None

    def load_markdown_files(self) -> List[str]:
        """Read every Markdown document in the data directory."""
        documents: List[str] = []
        if not os.path.isdir(self.data_dir):
            return documents
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith(".md"):
                path = os.path.join(self.data_dir, filename)
                with open(path, "r", encoding="utf-8") as handle:
                    documents.append(handle.read())
        return documents

    def process_documents(self) -> int:
        """Load, chunk and index the corpus. Returns the number of chunks."""
        from langchain_community.vectorstores import Chroma

        logger.info("Loading documents from %s", self.data_dir)
        documents = self.load_markdown_files()

        chunks: List[str] = []
        for document in documents:
            chunks.extend(self.text_splitter.split_text(document))
        logger.info("Created %d chunks", len(chunks))

        if not chunks:
            # Chroma.from_texts fails on an empty list; an empty corpus is a
            # normal state before the first upload.
            self.db = None
            logger.info("No documents indexed yet")
            return 0

        config.CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)
        self.db = Chroma.from_texts(
            chunks, self.embeddings, persist_directory=str(config.CHROMA_DIR)
        )
        logger.info("Vector store ready at %s", config.CHROMA_DIR)
        return len(chunks)

    def search(self, query: str, k: int = 3) -> List[Any]:
        if not self.db:
            raise ValueError("No documents have been indexed yet.")
        return self.db.similarity_search(query, k=k)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(f"Device: {_DEVICE.describe()}")

    rag = RAGSystem()
    if rag.process_documents() == 0:
        print(f"No .md files found in {rag.data_dir}. Process a PDF first.")
        return

    while True:
        try:
            query = input("\nQuery ('q' to quit): ")
        except EOFError:
            break
        if query.lower() == "q":
            break
        for i, doc in enumerate(rag.search(query), 1):
            print(f"\n--- Result {i} ---")
            print(doc.page_content)


if __name__ == "__main__":
    main()
