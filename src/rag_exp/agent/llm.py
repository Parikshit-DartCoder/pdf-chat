"""Shared chat-LLM factory pointing at Ollama."""
from __future__ import annotations

from langchain_ollama import ChatOllama

from ..config.settings import get_settings


def build_chat_llm(temperature: float = 0.1, timeout: float = 600.0) -> ChatOllama:
    """Build a chat LLM pointing at Ollama. `timeout` is the per-request HTTP
    timeout in seconds; the default of 600 covers long contexts on a shared GPU.
    """
    s = get_settings()
    return ChatOllama(
        model=s.llm_model,
        base_url=s.ollama_base_url,
        temperature=temperature,
        num_ctx=8192,
        timeout=timeout,
    )
