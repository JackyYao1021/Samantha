"""Configuration kept separate from Samantha's shell-command model."""

from dataclasses import dataclass
import os
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    vl_url: str = "http://127.0.0.1:11434/v1"
    vl_model: str = "qwen2.5vl:7b"
    vl_key: str = "ollama"
    timeout: float = 180
    embedding_backend: str = "flagembedding"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    ollama_url: str = "http://127.0.0.1:11434"
    qdrant_url: str = ""
    qdrant_key: str = ""
    qdrant_path: str = str(WORK_DIR / ".content-index" / "qdrant")
    collection: str = "samantha_content_flagembedding_v1"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    batch_size: int = 8
    pdf_vision: str = "auto"

    def __post_init__(self):
        if self.embedding_backend not in {"flagembedding", "ollama"}:
            raise ValueError("CONTENT_EMBEDDING_BACKEND must be flagembedding or ollama.")
        if self.pdf_vision not in {"auto", "always", "never"}:
            raise ValueError("CONTENT_PDF_VISION must be auto, always, or never.")
        if not 0 <= self.chunk_overlap < self.chunk_size or self.chunk_size > 6000:
            raise ValueError("Require 0 <= chunk_overlap < chunk_size <= 6000 (characters).")
        if self.batch_size < 1 or self.timeout <= 0:
            raise ValueError("batch_size and timeout must be positive.")

    @classmethod
    def from_env(cls):
        backend = os.getenv("CONTENT_EMBEDDING_BACKEND", "flagembedding")
        return cls(
            vl_url=os.getenv("CONTENT_VL_BASE_URL", os.getenv("QWEN_BASE_URL", cls.vl_url)),
            vl_model=os.getenv("CONTENT_VL_MODEL", cls.vl_model),
            vl_key=os.getenv("CONTENT_VL_API_KEY", "ollama"),
            timeout=float(os.getenv("CONTENT_TIMEOUT", "180")),
            embedding_backend=backend,
            embedding_model=os.getenv("CONTENT_EMBEDDING_MODEL") or ("bge-m3" if backend == "ollama" else "BAAI/bge-m3"),
            embedding_device=os.getenv("CONTENT_EMBEDDING_DEVICE", "cpu"),
            ollama_url=os.getenv("CONTENT_OLLAMA_URL", cls.ollama_url),
            qdrant_url=os.getenv("CONTENT_QDRANT_URL", ""),
            qdrant_key=os.getenv("CONTENT_QDRANT_API_KEY", ""),
            qdrant_path=os.getenv("CONTENT_QDRANT_PATH", cls.qdrant_path),
            collection=os.getenv("CONTENT_COLLECTION") or f"samantha_content_{backend}_v1",
            chunk_size=int(os.getenv("CONTENT_CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CONTENT_CHUNK_OVERLAP", "150")),
            batch_size=int(os.getenv("CONTENT_BATCH_SIZE", "8")),
            pdf_vision=os.getenv("CONTENT_PDF_VISION", "auto"),
        )
