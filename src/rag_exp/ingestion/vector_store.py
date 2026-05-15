"""Qdrant collection setup + bulk upsert of chunks.

Schema: a single collection with TWO named vector fields:
  - "dense"  : BGE-M3 cosine-similarity dense embedding (1024-dim)
  - "sparse" : BM25-style sparse vector (term-frequency, hash-tokenized);
               configured with Modifier.IDF so Qdrant scores it server-side
               as full BM25 at query time.

Old collections (anonymous single dense vector) are auto-detected and
recreated on next ingest -- the user re-ingests, no manual ops.
"""
from __future__ import annotations

import uuid
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..config.settings import get_settings
from ..retrieval import bm25
from .chunker import Chunk
from .embedder import build_embedder


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def _make_client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=60)


def _collection_has_hybrid_schema(client: QdrantClient, name: str) -> bool:
    """Return True iff the collection exists and already has both named
    dense + sparse vectors. Old single-vector collections return False."""
    try:
        info = client.get_collection(name)
    except Exception:
        return False
    cfg = info.config.params
    vectors = getattr(cfg, "vectors", None)
    sparse = getattr(cfg, "sparse_vectors", None)
    # Named-vector collections expose `vectors` as a dict of name -> params.
    return (
        isinstance(vectors, dict)
        and DENSE_VECTOR_NAME in vectors
        and isinstance(sparse, dict)
        and SPARSE_VECTOR_NAME in sparse
    )


def ensure_collection() -> QdrantClient:
    s = get_settings()
    client = _make_client()
    existing = {c.name for c in client.get_collections().collections}

    if s.qdrant_collection in existing and not _collection_has_hybrid_schema(client, s.qdrant_collection):
        # Old single-vector schema -- drop and recreate. Re-ingest required.
        client.delete_collection(s.qdrant_collection)
        existing.discard(s.qdrant_collection)

    if s.qdrant_collection not in existing:
        client.create_collection(
            collection_name=s.qdrant_collection,
            vectors_config={
                DENSE_VECTOR_NAME: qmodels.VectorParams(
                    size=s.embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                    modifier=qmodels.Modifier.IDF,
                ),
            },
        )
        client.create_payload_index(
            collection_name=s.qdrant_collection,
            field_name="source_path",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=s.qdrant_collection,
            field_name="language",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
    return client


def _chunk_id(chunk: Chunk) -> str:
    # Qdrant point IDs must be an unsigned int or a UUID string. UUID5 over
    # normalized (source, text) ONLY -- deliberately excluding page_number /
    # chunk_index. Repeated boilerplate (running headers, per-page credit
    # lines from OCR'd books) otherwise produced one point per page: a single
    # footer appeared 125x, crowding retrieval and flat-lining rerank scores.
    # Same text in the same doc now collapses to one representative point.
    norm = " ".join((chunk.text or "").split()).lower()
    raw = f"{chunk.source_path}|{norm}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _is_embeddable(text: str) -> bool:
    """Reject degenerate text that makes the embedding model emit NaN:
    empty/whitespace, too-short, or no alphanumeric/Arabic content."""
    t = (text or "").strip()
    if len(t) < 10:
        return False
    import re
    return bool(re.search(r"[0-9A-Za-z؀-ۿ]", t))


def _embed_one(embedder, text: str):
    """Embed a single text; return the vector or None if the model/server
    chokes (Ollama returns HTTP 500 'unsupported value: NaN' on bad input)."""
    try:
        return embedder.embed_documents([text])[0]
    except Exception:
        return None


def upsert_chunks(chunks: Iterable[Chunk], batch_size: int | None = None) -> int:
    s = get_settings()
    batch_size = batch_size or s.embed_batch
    client = ensure_collection()
    embedder = build_embedder()

    # Drop degenerate chunks up front so one bad page can't poison a batch.
    chunks = [c for c in chunks if _is_embeddable(c.text)]
    written = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            dense_vectors = embedder.embed_documents([c.text for c in batch])
        except Exception:
            # A single chunk in the batch made the embedder 500. Fall back to
            # per-chunk embedding and skip only the offending ones.
            dense_vectors = [_embed_one(embedder, c.text) for c in batch]

        points = []
        for c, dv in zip(batch, dense_vectors):
            if dv is None:
                continue  # unembeddable chunk -- skip, keep the rest
            indices, values = bm25.encode(c.text)
            points.append(
                qmodels.PointStruct(
                    id=_chunk_id(c),
                    vector={
                        DENSE_VECTOR_NAME: dv,
                        SPARSE_VECTOR_NAME: qmodels.SparseVector(
                            indices=indices, values=values,
                        ),
                    },
                    payload={
                        "text": c.text,
                        "source_path": c.source_path,
                        "page_number": c.page_number,
                        "chunk_index": c.chunk_index,
                        "language": c.language,
                    },
                )
            )

        if points:
            client.upsert(collection_name=s.qdrant_collection, points=points)
            written += len(points)
    return written
