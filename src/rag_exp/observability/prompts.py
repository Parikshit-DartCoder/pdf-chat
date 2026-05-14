"""Prompt registry. Loads from Langfuse when configured; otherwise falls back to
filesystem prompts under prompts/. The filesystem copy doubles as the
'initialize prompts' source-of-truth used to seed Langfuse on first run."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .langfuse_client import get_langfuse
from ..config.settings import get_settings


PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts"


def _load_fs_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found on disk: {path}")
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=32)
def get_prompt(name: str) -> str:
    """Return a prompt template by name. Tries Langfuse first."""
    s = get_settings()
    if s.langfuse_enabled:
        try:
            lf_prompt = get_langfuse().get_prompt(name)
            if lf_prompt is not None:
                return lf_prompt.prompt  # langfuse Prompt object
        except Exception:
            pass
    return _load_fs_prompt(name)


def seed_langfuse_prompts() -> int:
    """Push the prompts/*.md files into Langfuse so they're version-controlled
    centrally. Returns the number of prompts pushed."""
    s = get_settings()
    if not s.langfuse_enabled:
        return 0
    lf = get_langfuse()
    count = 0
    for md in PROMPT_DIR.glob("*.md"):
        text = md.read_text(encoding="utf-8")
        lf.create_prompt(name=md.stem, prompt=text, labels=["production"])
        count += 1
    return count
