"""Hash-tokenized BM25 sparse encoder.

Qdrant supports server-side BM25 ranking when a sparse vector collection is
configured with `Modifier.IDF` -- it tracks document frequencies internally
and applies IDF weighting at query time. All we need to provide is a sparse
vector of `(token_id, term_frequency)` per chunk.

This module produces those sparse vectors with no extra dependencies:
- Tokenize on Unicode word characters + the Arabic block (U+0600-U+06FF).
- Hash each token to a stable uint32 via BLAKE2b-4. Collisions on a 4-byte
  space are negligible for typical corpora (<1M unique terms).
- Aggregate to (indices, values) where values are raw term frequencies.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[\w؀-ۿ]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase, language-aware tokenization. Keeps English words and Arabic
    script segments; drops punctuation."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _stable_hash(token: str) -> int:
    """Stable uint32 hash. Stable across processes and across Python runs."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def encode(text: str) -> tuple[list[int], list[float]]:
    """Encode text as a sparse `(indices, values)` pair suitable for a Qdrant
    sparse vector with `Modifier.IDF`."""
    counts = Counter(tokenize(text))
    if not counts:
        # Qdrant rejects empty sparse vectors; emit a sentinel so the point
        # still upserts and is reachable by dense search alone.
        return [0], [0.0]
    indices: list[int] = []
    values: list[float] = []
    for tok, n in counts.items():
        indices.append(_stable_hash(tok))
        values.append(float(n))
    return indices, values
