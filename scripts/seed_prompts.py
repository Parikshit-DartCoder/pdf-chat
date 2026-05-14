"""One-shot helper: push prompts/*.md into Langfuse so they can be edited there."""
from __future__ import annotations

from rag_exp.observability.prompts import seed_langfuse_prompts


if __name__ == "__main__":
    pushed = seed_langfuse_prompts()
    print(f"pushed {pushed} prompts to Langfuse")
