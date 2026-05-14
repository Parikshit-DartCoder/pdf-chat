"""Ollama-served embeddings. BGE-M3 is multilingual (handles EN + AR) and 1024-dim."""
from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from ..config.settings import get_settings


def build_embedder() -> OllamaEmbeddings:
    s = get_settings()
    return OllamaEmbeddings(model=s.embedding_model, base_url=s.ollama_base_url)
