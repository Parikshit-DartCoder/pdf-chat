"""Central typed config. Resolution order for every value:

    1. environment variable (e.g. TOP_K=15) -- highest priority, for one-off tweaks
    2. configs/retrieval.toml -- versioned, the source of truth for retrieval knobs
    3. built-in default below                -- safety net

This lets you commit a tuned retrieval profile and still override any single
value at runtime without editing files.
"""
from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Path resolution: $RAG_RETRIEVAL_CONFIG > /app/configs/retrieval.toml > repo-local fallback.
_DEFAULT_CONFIG_PATH = Path("/app/configs/retrieval.toml")


def _load_toml() -> dict[str, Any]:
    path_str = os.environ.get("RAG_RETRIEVAL_CONFIG")
    path = Path(path_str) if path_str else _DEFAULT_CONFIG_PATH
    if not path.exists():
        # Try a repo-relative location as a dev convenience.
        alt = Path(__file__).resolve().parents[3] / "configs" / "retrieval.toml"
        if alt.exists():
            path = alt
        else:
            log.info("retrieval config not found at %s; using built-in defaults", path)
            return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception as e:
        log.warning("failed to parse %s: %s; falling back to defaults", path, e)
        return {}


def _env(key: str, default: str | None = None, required: bool = False) -> str:
    val = os.environ.get(key, default)
    if required and not val:
        raise RuntimeError(f"missing required env var: {key}")
    return val  # type: ignore[return-value]


def _from(
    env_key: str,
    cfg: dict[str, Any],
    section: str,
    field: str,
    default: Any,
    cast=str,
) -> Any:
    """Resolve a single value with env > cfg > default precedence and cast it."""
    raw = os.environ.get(env_key)
    if raw is None:
        raw = cfg.get(section, {}).get(field, default)
    if cast is bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).lower() in ("1", "true", "yes", "on")
    if cast is str:
        return str(raw)
    return cast(raw)


@dataclass(frozen=True)
class Settings:
    # ---- LLM / embeddings (Ollama) ----
    ollama_base_url: str
    llm_model: str
    embedding_model: str
    reranker_model: str

    # ---- Vector store (Qdrant) ----
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    embedding_dim: int

    # ---- Ingestion ----
    pdf_input_dir: str
    chunk_size: int
    chunk_overlap: int
    embed_batch: int
    ocr_max_concurrency: int

    # ---- Memory ----
    memory_db_path: str

    # ---- Retrieval ----
    top_k: int
    rerank_top_n: int
    hybrid_enabled: bool
    sparse_prefetch_k: int
    dense_prefetch_k: int
    rerank_score_floor: float
    rerank_min_chunks: int       # if fewer than this clear the floor, ignore the floor
    rerank_instruction: str

    # ---- Observability (Langfuse) ----
    langfuse_host: str
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_enabled: bool

    # ---- OCR (PaddleOCR-VL via vLLM) ----
    paddleocr_vl_url: str
    paddleocr_vl_model: str
    paddleocr_vl_timeout_s: int
    ocr_min_chars_per_page: int

    # ---- App ----
    app_title: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    cfg = _load_toml()

    return Settings(
        # Infra and identity are env-only -- the TOML is for tuning, not topology.
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://ollama:11434"),
        llm_model=_env("LLM_MODEL", "qwen2.5:7b-instruct"),
        embedding_model=_env("EMBEDDING_MODEL", "bge-m3"),
        reranker_model=_env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        qdrant_url=_env("QDRANT_URL", "http://qdrant:6333"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
        qdrant_collection=_env("QDRANT_COLLECTION", "rag_exp_docs"),
        embedding_dim=int(_env("EMBEDDING_DIM", "1024")),
        pdf_input_dir=_env("PDF_INPUT_DIR", "/app/data/pdfs"),
        memory_db_path=_env("MEMORY_DB_PATH", "/app/data/memory.sqlite"),
        langfuse_host=_env("LANGFUSE_HOST", "http://langfuse:3000"),
        langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        langfuse_enabled=bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")),
        paddleocr_vl_url=_env("PADDLEOCR_VL_URL", "http://paddleocr-vl:8118/v1"),
        paddleocr_vl_model=_env("PADDLEOCR_VL_MODEL", "PaddleOCR-VL-1.5-0.9B"),
        paddleocr_vl_timeout_s=int(_env("PADDLEOCR_VL_TIMEOUT_S", "120")),
        app_title=_env("APP_TITLE", "PDF Chat"),

        # Retrieval / ingestion tunables -- resolved env > TOML > default.
        chunk_size=_from("CHUNK_SIZE", cfg, "chunking", "chunk_size", 1200, int),
        chunk_overlap=_from("CHUNK_OVERLAP", cfg, "chunking", "chunk_overlap", 150, int),
        embed_batch=_from("EMBED_BATCH", cfg, "ingestion", "embed_batch", 256, int),
        ocr_max_concurrency=_from("OCR_MAX_CONCURRENCY", cfg, "ingestion", "ocr_max_concurrency", 8, int),
        ocr_min_chars_per_page=_from("OCR_MIN_CHARS_PER_PAGE", cfg, "ingestion", "ocr_min_chars_per_page", 40, int),

        top_k=_from("TOP_K", cfg, "retrieval", "top_k", 30, int),
        rerank_top_n=_from("RERANK_TOP_N", cfg, "retrieval", "rerank_top_n", 7, int),
        hybrid_enabled=_from("HYBRID_ENABLED", cfg, "retrieval", "hybrid_enabled", True, bool),
        sparse_prefetch_k=_from("SPARSE_PREFETCH_K", cfg, "retrieval", "sparse_prefetch_k", 60, int),
        dense_prefetch_k=_from("DENSE_PREFETCH_K", cfg, "retrieval", "dense_prefetch_k", 60, int),
        rerank_score_floor=_from("RERANK_SCORE_FLOOR", cfg, "retrieval", "rerank_score_floor", 0.10, float),
        rerank_min_chunks=_from("RERANK_MIN_CHUNKS", cfg, "retrieval", "rerank_min_chunks", 3, int),
        rerank_instruction=_from(
            "RERANK_INSTRUCTION", cfg, "retrieval", "rerank_instruction",
            "Given a question, retrieve passages that contain the answer.", str,
        ),
    )
