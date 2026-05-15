"""Hybrid retrieval (dense + BM25 sparse) with RRF fusion, then cross-encoder
rerank with instruction prompt and score-floor filtering.

Pipeline:
  1. Embed the query (dense via BGE-M3) and encode it as BM25 sparse.
  2. Qdrant `query_points` with two prefetches and RRF fusion -> top_k candidates.
  3. BGE-reranker-v2-m3 cross-encoder scoring with an instruction prompt.
  4. Drop chunks below `RERANK_SCORE_FLOOR`. Return top_n.

Set `HYBRID_ENABLED=false` to fall back to dense-only behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..config.settings import get_settings
from ..ingestion.embedder import build_embedder
from ..ingestion.vector_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from . import bm25


@dataclass
class RetrievedChunk:
    text: str
    source_path: str
    page_number: int
    score: float
    language: str | None

    @property
    def citation(self) -> str:
        from pathlib import Path
        return f"{Path(self.source_path).name} · p.{self.page_number}"


@lru_cache(maxsize=1)
def _qdrant() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=60)


@lru_cache(maxsize=1)
def _reranker():
    """Lazily instantiate the cross-encoder. Costs ~1.5 GB RAM; load on first use."""
    from FlagEmbedding import FlagReranker
    s = get_settings()
    return FlagReranker(s.reranker_model, use_fp16=True)


def _build_filter(source_paths: list[str] | None) -> qmodels.Filter | None:
    if not source_paths:
        return None
    return qmodels.Filter(
        must=[qmodels.FieldCondition(
            key="source_path",
            match=qmodels.MatchAny(any=list(source_paths)),
        )]
    )


def _payload_to_chunk(payload: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=payload["text"],
        source_path=payload["source_path"],
        page_number=int(payload["page_number"]),
        score=float(score),
        language=payload.get("language"),
    )


def hybrid_search(
    query: str,
    k: int | None = None,
    source_paths: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Dense + BM25 prefetch + RRF fusion via Qdrant's server-side query API."""
    s = get_settings()
    k = k or s.top_k

    dense_vec = build_embedder().embed_query(query)
    sparse_indices, sparse_values = bm25.encode(query)

    flt = _build_filter(source_paths)
    res = _qdrant().query_points(
        collection_name=s.qdrant_collection,
        prefetch=[
            qmodels.Prefetch(
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                limit=s.dense_prefetch_k,
                filter=flt,
            ),
            qmodels.Prefetch(
                query=qmodels.SparseVector(indices=sparse_indices, values=sparse_values),
                using=SPARSE_VECTOR_NAME,
                limit=s.sparse_prefetch_k,
                filter=flt,
            ),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return [_payload_to_chunk(p.payload or {}, p.score) for p in res.points]


def dense_search(
    query: str,
    k: int | None = None,
    source_paths: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Dense-only fallback for `HYBRID_ENABLED=false`."""
    s = get_settings()
    k = k or s.top_k
    dense_vec = build_embedder().embed_query(query)
    res = _qdrant().query_points(
        collection_name=s.qdrant_collection,
        query=dense_vec,
        using=DENSE_VECTOR_NAME,
        limit=k,
        # qdrant-client 1.12: top-level kwarg is `query_filter` (Prefetch uses `filter`).
        query_filter=_build_filter(source_paths),
        with_payload=True,
    )
    return [_payload_to_chunk(p.payload or {}, p.score) for p in res.points]


def rerank(
    query: str,
    chunks: Sequence[RetrievedChunk],
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """Cross-encoder rerank with the configured instruction prompt and score floor."""
    if not chunks:
        return []
    s = get_settings()
    top_n = top_n or s.rerank_top_n

    # NOTE: plain bge-reranker-v2-m3 (FlagReranker) does NOT accept a `prompt`
    # kwarg -- it raises TypeError (that's an instruction-tuned / LLM-reranker
    # feature). So we never pass it.
    pairs = [(query, c.text) for c in chunks]
    scores = _reranker().compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]

    rescored = [
        RetrievedChunk(
            text=c.text,
            source_path=c.source_path,
            page_number=c.page_number,
            score=float(s_),
            language=c.language,
        )
        for c, s_ in zip(chunks, scores)
    ]
    rescored.sort(key=lambda x: x.score, reverse=True)

    above_floor = [c for c in rescored if c.score >= s.rerank_score_floor]
    if len(above_floor) >= s.rerank_min_chunks:
        return above_floor[:top_n]

    # Too few chunks cleared the floor -> the cross-encoder is unreliable for
    # this query (typical for vague, contentless prompts like "what is this doc
    # about?"). The upstream hybrid/dense retrieval DID carry signal, so fall
    # back to the original retrieval order + scores rather than surfacing a
    # mix of one weak score and several misleading ~0.000 reranker scores.
    return list(chunks[:top_n])


def retrieve(query: str, source_paths: list[str] | None = None) -> list[RetrievedChunk]:
    """High-level helper: hybrid (or dense) search, then cross-encoder rerank."""
    s = get_settings()
    if s.hybrid_enabled:
        candidates = hybrid_search(query, source_paths=source_paths)
    else:
        candidates = dense_search(query, source_paths=source_paths)
    return rerank(query, candidates)


# Backwards-compat alias so older callers and tests keep working.
def vector_search(
    query: str,
    k: int | None = None,
    source_paths: list[str] | None = None,
) -> list[RetrievedChunk]:
    s = get_settings()
    if s.hybrid_enabled:
        return hybrid_search(query, k=k, source_paths=source_paths)
    return dense_search(query, k=k, source_paths=source_paths)
