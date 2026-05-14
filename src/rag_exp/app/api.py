"""FastAPI backend for rag_exp.

Endpoint surface (all under /v1 unless marked):

    GET  /healthz                        liveness probe
    GET  /v1/health/services             per-dependency status (qdrant, ollama, paddleocr-vl, langfuse)
    POST /v1/sessions                    create a new conversation session id
    POST /v1/chat                        synchronous turn -> {answer, citations, route}
    POST /v1/chat/stream                 same, but SSE (single answer chunk; per-token streaming TBD)
    POST /v1/ingest                      multipart PDF upload -> parse, chunk, embed, upsert
    GET  /v1/sessions/{id}/documents     docs scoped to this session
    GET  /v1/corpus                      every source_path indexed (any session)
    POST /v1/chat/completions            OpenAI-compatible shim so OpenWebUI works unchanged

The service is process-isolated from Streamlit. Streamlit + OpenWebUI both
talk to it over HTTP -- no in-process imports.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from rag_exp.agent.graph import run_agent
from rag_exp.config.settings import get_settings
from rag_exp.ingestion.chunker import chunk_pages
from rag_exp.ingestion.parser import parse_pdf
from rag_exp.ingestion.vector_store import upsert_chunks
from rag_exp.memory import conversation as mem


app = FastAPI(
    title="PDF Chat API",
    version="0.1.0",
    description="REST backend for the PDF Chat agentic RAG system.",
)

# OpenWebUI sends requests from its own origin; allow CORS broadly so any UI works.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SessionOut(BaseModel):
    session_id: str


class ChatIn(BaseModel):
    session_id: str
    question: str


class Citation(BaseModel):
    index: int
    source: str
    page: int
    score: float
    snippet: str


class ChatOut(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: str


class IngestedFile(BaseModel):
    name: str
    pages: int
    chunks: int


class IngestOut(BaseModel):
    session_id: str
    files: list[IngestedFile]


class SessionDocOut(BaseModel):
    source_path: str
    display_name: str


class HealthOut(BaseModel):
    qdrant: bool
    ollama: bool
    paddleocr_vl: bool
    langfuse: bool


# OpenAI-compatible types (minimal — just what OpenWebUI needs)
class OAMessage(BaseModel):
    role: str
    content: Any   # string or list[parts]


class OAChatIn(BaseModel):
    model: str | None = None
    messages: list[OAMessage]
    stream: bool = False
    user: str | None = None   # used as session_id when present


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _on_startup() -> None:
    mem.init_memory()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/v1/health/services", response_model=HealthOut)
async def health_services() -> HealthOut:
    s = get_settings()

    async def _probe(url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(url)
                return r.status_code < 500
        except Exception:
            return False

    qdrant_ok, ollama_ok, paddle_ok = await asyncio.gather(
        _probe(f"{s.qdrant_url}/collections"),
        _probe(f"{s.ollama_base_url}/api/tags"),
        _probe(f"{s.paddleocr_vl_url}/models"),
    )
    langfuse_ok = await _probe(f"{s.langfuse_host}/api/public/health")
    return HealthOut(
        qdrant=qdrant_ok, ollama=ollama_ok,
        paddleocr_vl=paddle_ok, langfuse=langfuse_ok,
    )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@app.post("/v1/sessions", response_model=SessionOut)
def new_session() -> SessionOut:
    return SessionOut(session_id=str(uuid.uuid4()))


@app.get("/v1/sessions/{session_id}/documents", response_model=list[SessionDocOut])
def session_documents(session_id: str) -> list[SessionDocOut]:
    docs = mem.list_session_documents(session_id)
    return [SessionDocOut(source_path=d.source_path, display_name=d.display_name) for d in docs]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

@app.get("/v1/corpus")
def corpus() -> dict:
    try:
        paths = mem.list_corpus_documents()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"corpus listing failed: {e}")
    return {"documents": paths}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.post("/v1/chat", response_model=ChatOut)
def chat(req: ChatIn) -> ChatOut:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must be non-empty")
    result = run_agent(question=req.question, session_id=req.session_id)
    return ChatOut(
        answer=result.answer,
        citations=[Citation(**c) for c in result.citations],
        route=result.route,
    )


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatIn):
    """SSE: emits one `delta` event with the full answer, then a `done` event
    with citations and route. Per-token streaming is a follow-up — would
    require make_answer to expose a streaming API."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must be non-empty")

    async def _gen():
        # Run the synchronous agent in a thread so we don't block the event loop.
        result = await asyncio.to_thread(run_agent, req.question, req.session_id)
        yield {"event": "delta", "data": json.dumps({"answer": result.answer})}
        yield {"event": "done", "data": json.dumps({
            "citations": result.citations, "route": result.route,
        })}

    return EventSourceResponse(_gen())


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@app.post("/v1/ingest", response_model=IngestOut)
async def ingest(
    session_id: str = Form(...),
    files: list[UploadFile] = File(...),
) -> IngestOut:
    s = get_settings()
    pdf_dir = Path(s.pdf_input_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    mem.init_memory()

    out: list[IngestedFile] = []
    for f in files:
        dest = pdf_dir / f.filename
        dest.write_bytes(await f.read())

        # Heavy work in a thread; the API stays responsive.
        def _process(p: Path = dest) -> tuple[int, int]:
            pages = parse_pdf(p)
            chunks = chunk_pages(pages, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
            written = upsert_chunks(chunks)
            return len(pages), written

        n_pages, n_chunks = await asyncio.to_thread(_process)
        mem.add_session_document(session_id, str(dest), f.filename)
        out.append(IngestedFile(name=f.filename, pages=n_pages, chunks=n_chunks))

    return IngestOut(session_id=session_id, files=out)


# ---------------------------------------------------------------------------
# OpenAI-compatible shim (for OpenWebUI and other off-the-shelf clients)
# ---------------------------------------------------------------------------

def _extract_text(content: Any) -> str:
    """OpenAI messages.content may be a string OR a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return " ".join(parts).strip()
    return str(content)


def _last_user_message(messages: list[OAMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return _extract_text(m.content)
    return ""


@app.post("/v1/chat/completions")
async def openai_chat_completions(req: OAChatIn):
    question = _last_user_message(req.messages)
    if not question:
        raise HTTPException(status_code=400, detail="no user message found")

    # Stable per-user session so multi-turn works inside OpenWebUI.
    session_id = req.user or "openwebui-default"

    result = await asyncio.to_thread(run_agent, question, session_id)
    created = int(time.time())
    model_name = req.model or "pdf-chat"

    # Append citations as a markdown footer; OpenWebUI shows them inline.
    answer = result.answer
    if result.citations:
        lines = ["", "---", "**Sources:**"]
        for c in result.citations:
            name = Path(c["source"]).name
            lines.append(f"- [{c['index']}] {name} · p.{c['page']} · score {c['score']:.3f}")
        answer = f"{answer}\n" + "\n".join(lines)

    if not req.stream:
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # Stream as a single chunk + done sentinel. Good enough for OpenWebUI.
    async def _stream():
        chunk_id = f"chatcmpl-{uuid.uuid4()}"
        first = {
            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer},
                         "finish_reason": None}],
        }
        yield {"data": json.dumps(first)}
        done = {
            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield {"data": json.dumps(done)}
        yield {"data": "[DONE]"}

    return EventSourceResponse(_stream())


# Tiny info endpoint OpenWebUI sometimes hits.
@app.get("/v1/models")
def list_models() -> dict:
    s = get_settings()
    return {
        "object": "list",
        "data": [{"id": "pdf-chat", "object": "model", "owned_by": "rag_exp",
                  "permission": [], "root": s.llm_model, "parent": None}],
    }
