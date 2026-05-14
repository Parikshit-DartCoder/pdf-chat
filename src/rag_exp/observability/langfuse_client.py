"""Thin wrapper around Langfuse: gracefully no-ops if keys are absent so the
app still runs in 'offline' mode for local development."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..config.settings import get_settings


class _NoopLangfuse:
    """Drop-in placeholder when Langfuse is disabled."""

    def trace(self, **_: Any):
        return self

    def span(self, **_: Any):
        return self

    def generation(self, **_: Any):
        return self

    def event(self, **_: Any):
        return self

    def end(self, **_: Any):
        return None

    def update(self, **_: Any):
        return None

    def get_prompt(self, name: str, **_: Any):
        return None

    def flush(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@lru_cache(maxsize=1)
def get_langfuse():
    s = get_settings()
    if not s.langfuse_enabled:
        return _NoopLangfuse()
    from langfuse import Langfuse

    return Langfuse(
        host=s.langfuse_host,
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
    )


def get_callback_handler(session_id: str | None = None, user_id: str | None = None):
    """Return a LangChain callback handler that streams traces to Langfuse."""
    s = get_settings()
    if not s.langfuse_enabled:
        return None
    from langfuse.callback import CallbackHandler

    return CallbackHandler(
        host=s.langfuse_host,
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        session_id=session_id,
        user_id=user_id,
    )
