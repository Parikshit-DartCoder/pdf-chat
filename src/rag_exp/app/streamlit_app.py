"""Streamlit chatbot UI. Pure HTTP client of the rag_exp REST API.

No in-process imports of `rag_exp.agent`, `rag_exp.ingestion`, or
`rag_exp.memory` -- everything goes through `RAG_API_URL`.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Iterator

import httpx
import streamlit as st


API_URL = os.environ.get("RAG_API_URL", "http://api:8000")
APP_TITLE = os.environ.get("APP_TITLE", "PDF Chat")
CHAT_TIMEOUT_S = float(os.environ.get("CHAT_TIMEOUT_S", "600"))
INGEST_TIMEOUT_S = float(os.environ.get("INGEST_TIMEOUT_S", "1800"))


def _http() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=CHAT_TIMEOUT_S)


def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        try:
            with _http() as c:
                st.session_state.session_id = c.post("/v1/sessions").json()["session_id"]
        except Exception:
            st.session_state.session_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []
    if "ingesting" not in st.session_state:
        st.session_state.ingesting = False


_STAGE_TEXT = {
    "saved": "Upload received",
    "pypdf": "Extracting embedded text",
    "docling": "Layout-aware extraction (Docling)",
    "rasterize": "Rendering scanned pages for OCR",
    "ocr": "OCR (PaddleOCR-VL)",
    "parsed": "Parsing complete",
    "chunking": "Chunking",
    "indexing": "Embedding + indexing",
}


def _stage_label(data: dict) -> str:
    stage = data.get("stage", "")
    base = _STAGE_TEXT.get(stage, stage)
    if stage == "ocr" and "total" in data:
        return f"{base} — page {data.get('done', 0)}/{data['total']}"
    if stage in ("chunking", "indexing") and data.get("total") is not None:
        unit = "pages" if stage == "chunking" else "chunks"
        return f"{base} — {data['total']} {unit}"
    return base


def _stream_ingest(files, session_id: str) -> Iterator[tuple[str, dict]]:
    """Yield (event, data) tuples from the streaming ingest endpoint so the UI
    can render each stage live and keep chat locked until `done`."""
    payload_files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
    with httpx.Client(base_url=API_URL, timeout=INGEST_TIMEOUT_S) as c:
        with c.stream(
            "POST", "/v1/ingest/stream",
            data={"session_id": session_id}, files=payload_files,
        ) as r:
            r.raise_for_status()
            event = "message"
            for line in r.iter_lines():
                if not line:
                    event = "message"
                    continue
                if line.startswith("event:"):
                    event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    raw = line[len("data:"):].strip()
                    try:
                        yield event, json.loads(raw)
                    except json.JSONDecodeError:
                        continue


def _post_chat(session_id: str, question: str) -> dict:
    with _http() as c:
        r = c.post("/v1/chat", json={"session_id": session_id, "question": question})
        r.raise_for_status()
        return r.json()


def _get_session_documents(session_id: str) -> list[dict]:
    try:
        with _http() as c:
            r = c.get(f"/v1/sessions/{session_id}/documents")
            r.raise_for_status()
            return r.json()
    except Exception:
        return []


def _get_corpus() -> list[str]:
    try:
        with _http() as c:
            r = c.get("/v1/corpus")
            r.raise_for_status()
            return r.json().get("documents", [])
    except Exception:
        return []


def _get_health() -> dict:
    try:
        with _http() as c:
            r = c.get("/v1/health/services")
            r.raise_for_status()
            return r.json()
    except Exception:
        return {}


def _render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for c in citations:
            name = Path(c["source"]).name
            st.markdown(
                f"**[{c['index']}] {name} · page {c['page']}** · score `{c['score']:.3f}`"
            )
            st.caption(c["snippet"] + ("..." if len(c["snippet"]) >= 240 else ""))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _init_session_state()

    with st.sidebar:
        st.title("PDF Chat")
        st.caption("Agentic RAG · FastAPI · LangGraph · Qdrant · Ollama")
        st.markdown(f"**API:** `{API_URL}`")
        st.markdown(f"**Session:** `{st.session_state.session_id[:8]}`")

        with st.expander("Service health"):
            for svc, ok in _get_health().items():
                st.markdown(f"- **{svc}**: {'up' if ok else 'down'}")

        if st.button("New session (resets document scope)", use_container_width=True):
            try:
                with _http() as c:
                    st.session_state.session_id = c.post("/v1/sessions").json()["session_id"]
            except Exception:
                st.session_state.session_id = str(uuid.uuid4())
            st.session_state.history = []
            st.rerun()
        st.caption(
            "A session groups documents. Every PDF you ingest in this session "
            "is searched together; starting a new session clears that scope."
        )

        st.divider()
        st.subheader("Add PDFs")
        uploaded = st.file_uploader(
            "Upload PDFs (EN/AR)",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        ingest_clicked = st.button(
            "Ingest uploaded",
            type="primary",
            use_container_width=True,
            disabled=not uploaded or st.session_state.ingesting,
        )
        if uploaded and ingest_clicked:
            st.session_state.ingesting = True
            with st.status("Indexing — chat is locked until this finishes",
                           expanded=True) as status:
                try:
                    last_stage = None
                    for event, data in _stream_ingest(
                        uploaded, st.session_state.session_id
                    ):
                        if event == "stage":
                            label = _stage_label(data)
                            # Collapse repeated OCR ticks into one updating line.
                            if data.get("stage") == "ocr":
                                status.update(label=f"{data['file']}: {label}")
                            elif label != last_stage:
                                st.write(f"- {data['file']}: {label}")
                                last_stage = label
                        elif event == "file":
                            st.write(
                                f"**{data['name']}** indexed: "
                                f"{data['pages']} pages, {data['chunks']} chunks"
                            )
                        elif event == "error":
                            raise RuntimeError(data.get("message", "ingestion failed"))
                    status.update(label="Indexing complete — chat unlocked",
                                  state="complete")
                except Exception as e:
                    status.update(label="Ingestion failed", state="error")
                    st.exception(e)
                finally:
                    st.session_state.ingesting = False
            st.rerun()

        session_docs = _get_session_documents(st.session_state.session_id)
        if session_docs:
            with st.expander(f"Searched in this session ({len(session_docs)})", expanded=True):
                for d in session_docs:
                    st.caption(d["display_name"])
                st.caption(
                    "Answers are scoped to ALL of these documents. Add more and "
                    "they join the same scope; start a new session to reset."
                )
        else:
            st.warning(
                "No documents in this session yet — questions will search the "
                "ENTIRE corpus (every file ever ingested, including other "
                "sessions and CLI bulk loads). Upload a PDF to scope answers "
                "to just your documents."
            )

        with st.expander("Full corpus index"):
            corpus = _get_corpus()
            if corpus:
                for p in corpus:
                    st.caption(Path(p).name)
            else:
                st.caption("Empty.")

    st.title(APP_TITLE)

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                _render_citations(turn.get("citations", []))

    if st.session_state.ingesting:
        st.info("Indexing in progress — chat is locked until all documents "
                "are parsed, embedded and indexed.")

    user_input = st.chat_input(
        "Indexing — please wait"
        if st.session_state.ingesting
        else "Ask a question about the indexed PDFs",
        disabled=st.session_state.ingesting,
    )
    if not user_input:
        return

    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            try:
                result = _post_chat(st.session_state.session_id, user_input)
            except httpx.HTTPError as e:
                st.error(f"Backend error: {e}")
                return
        st.markdown(result.get("answer") or "_(no answer)_")
        _render_citations(result.get("citations", []))

    st.session_state.history.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "citations": result.get("citations", []),
    })


if __name__ == "__main__":
    main()
