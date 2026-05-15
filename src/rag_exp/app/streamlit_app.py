"""Streamlit chatbot UI. Pure HTTP client of the rag_exp REST API.

No in-process imports of `rag_exp.agent`, `rag_exp.ingestion`, or
`rag_exp.memory` -- everything goes through `RAG_API_URL`.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

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


def _post_ingest(files, session_id: str) -> list[dict]:
    payload_files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
    with httpx.Client(base_url=API_URL, timeout=INGEST_TIMEOUT_S) as c:
        r = c.post("/v1/ingest", data={"session_id": session_id}, files=payload_files)
        r.raise_for_status()
        return r.json()["files"]


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

        if st.button("New conversation", use_container_width=True):
            try:
                with _http() as c:
                    st.session_state.session_id = c.post("/v1/sessions").json()["session_id"]
            except Exception:
                st.session_state.session_id = str(uuid.uuid4())
            st.session_state.history = []
            st.rerun()

        st.divider()
        st.subheader("Add PDFs")
        uploaded = st.file_uploader(
            "Upload PDFs (EN/AR)",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded and st.button("Ingest uploaded", type="primary", use_container_width=True):
            with st.status("Ingesting", expanded=True) as status:
                try:
                    results = _post_ingest(uploaded, st.session_state.session_id)
                    for r in results:
                        st.write(f"- {r['name']}: {r['pages']} pages, {r['chunks']} chunks")
                    status.update(label="Done", state="complete")
                except Exception as e:
                    status.update(label="Ingestion failed", state="error")
                    st.exception(e)

        session_docs = _get_session_documents(st.session_state.session_id)
        if session_docs:
            with st.expander(f"Active in this session ({len(session_docs)})", expanded=True):
                for d in session_docs:
                    st.caption(d["display_name"])
                st.caption("Retrieval is scoped to these documents.")
        else:
            st.caption("No session-scoped documents yet. Retrieval will search the full corpus.")

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

    user_input = st.chat_input("Ask a question about the indexed PDFs")
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
