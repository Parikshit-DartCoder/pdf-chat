"""Reusable node factories for the LangGraph agent.

Each factory returns a `(state) -> state` callable bound to its dependencies.
This lets you compose different graphs for different use cases — swap the LLM,
the prompt, the retriever, or the memory backend without rewriting the node.

Typical use:

    from rag_exp.agent.nodes import make_router, make_answer, MemoryBackend
    from rag_exp.agent.graph import build_graph

    graph = build_graph(
        nodes={
            "route": make_router(prompt_name="legal_router",
                                 allowed_routes=("retrieve", "escalate", "chitchat")),
            "answer": make_answer(llm_factory=lambda: ChatOpenAI(model="gpt-4o")),
        },
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..observability.prompts import get_prompt
from ..retrieval.retriever import RetrievedChunk, retrieve as default_retrieve
from .llm import build_chat_llm


# ---------------------------------------------------------------------------
# Dependency types
# ---------------------------------------------------------------------------

LLMFactory = Callable[[], Any]           # () -> a LangChain chat model
Retriever = Callable[[str], Sequence[RetrievedChunk]]
PromptFormatter = Callable[[Mapping[str, Any]], str]  # render the prompt body
CitationBuilder = Callable[[Sequence[RetrievedChunk]], list[dict]]
Node = Callable[[dict], dict]


@dataclass
class MemoryBackend:
    """Pluggable conversation memory. Default impl is SQLite under
    rag_exp.memory.conversation; you can supply Redis, Postgres, or in-memory."""

    load_summary: Callable[[str], str | None]
    save_summary: Callable[[str, str], None]
    append_turn: Callable[[str, str, str], None]   # (session_id, role, content)
    list_session_documents: Callable[[str], Sequence[str]]   # returns source_paths
    init: Callable[[], None] = lambda: None


def default_memory_backend() -> MemoryBackend:
    from ..memory import conversation as mem

    def _list_session_docs(session_id: str) -> list[str]:
        return [d.source_path for d in mem.list_session_documents(session_id)]

    return MemoryBackend(
        load_summary=mem.get_summary,
        save_summary=mem.set_summary,
        append_turn=mem.append_turn,
        list_session_documents=_list_session_docs,
        init=mem.init_memory,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invoke(llm: Any, prompt: str) -> str:
    msg = llm.invoke(prompt)
    return (getattr(msg, "content", None) or str(msg)).strip()


def _default_prompt_render(name: str, **fields: Any) -> str:
    return get_prompt(name).format(**fields)


def _default_format_excerpts(chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return "(no excerpts retrieved)"
    return "\n\n".join(f"[{i + 1}] ({c.citation})\n{c.text}" for i, c in enumerate(chunks))


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def make_load_memory(memory: MemoryBackend | None = None) -> Node:
    backend = memory or default_memory_backend()

    def node(state: dict) -> dict:
        sid = state["session_id"]
        state["summary"] = backend.load_summary(sid) or ""
        if "source_paths" not in state or not state.get("source_paths"):
            # If the caller did not pre-scope to specific docs, fall back to
            # whatever this session has registered.
            session_docs = list(backend.list_session_documents(sid))
            if session_docs:
                state["source_paths"] = session_docs
        return state

    return node


def make_router(
    llm_factory: LLMFactory = build_chat_llm,
    prompt_name: str = "router",
    allowed_routes: Sequence[str] = ("retrieve", "chitchat", "clarify"),
    default_route: str = "retrieve",
) -> Node:
    """LLM classifier. Returns one of `allowed_routes`; falls back to `default_route`."""

    def node(state: dict) -> dict:
        prompt = _default_prompt_render(
            prompt_name,
            summary=state.get("summary", ""),
            question=state["question"],
        )
        raw = _invoke(llm_factory(), prompt).lower().strip()
        chosen = default_route
        for label in allowed_routes:
            if label in raw:
                chosen = label
                break
        state["route"] = chosen
        return state

    return node


def make_rewrite_query(
    llm_factory: LLMFactory = build_chat_llm,
    prompt_name: str = "query_rewriter",
) -> Node:
    def node(state: dict) -> dict:
        prompt = _default_prompt_render(
            prompt_name,
            summary=state.get("summary", ""),
            question=state["question"],
        )
        rewritten = _invoke(llm_factory(), prompt)
        state["search_query"] = rewritten or state["question"]
        return state

    return node


def default_citation_builder(snippet_chars: int = 240) -> CitationBuilder:
    """Build the default citation shape: index, source, page, score, snippet."""

    def build(chunks: Sequence[RetrievedChunk]) -> list[dict]:
        return [
            {
                "index": i + 1,
                "source": c.source_path,
                "page": c.page_number,
                "score": c.score,
                "snippet": c.text[:snippet_chars],
            }
            for i, c in enumerate(chunks)
        ]

    return build


def make_retrieve(
    retriever: Retriever = default_retrieve,
    citation_builder: CitationBuilder | None = None,
    snippet_chars: int = 240,
    respect_source_paths: bool = True,
) -> Node:
    """Wrap any `(query, source_paths=None) -> list[RetrievedChunk]` retriever
    into a node.

    If `respect_source_paths` is True (default) and `state["source_paths"]` is
    non-empty, the retriever is called with that list as a filter so search is
    scoped to the session's documents. Falls back to corpus-wide search when
    no per-session scope exists.

    Citations are produced via `citation_builder`. The default builder returns
    `{index, source, page, score, snippet}` dicts. Supply your own to emit a
    different citation contract. `snippet_chars` is forwarded to the default
    builder; ignored when a custom `citation_builder` is provided.
    """
    builder = citation_builder or default_citation_builder(snippet_chars)

    def node(state: dict) -> dict:
        query = state.get("search_query") or state["question"]
        kwargs: dict[str, Any] = {}
        if respect_source_paths and state.get("source_paths"):
            kwargs["source_paths"] = list(state["source_paths"])
        try:
            chunks = list(retriever(query, **kwargs))
        except TypeError:
            # Retriever doesn't accept source_paths; call without it.
            chunks = list(retriever(query))
        state["retrieved"] = chunks
        state["citations"] = builder(chunks)
        return state

    return node


def make_answer(
    llm_factory: LLMFactory = build_chat_llm,
    prompt_name: str = "answerer",
    format_excerpts: Callable[[Sequence[RetrievedChunk]], str] = _default_format_excerpts,
) -> Node:
    def node(state: dict) -> dict:
        prompt = _default_prompt_render(
            prompt_name,
            context=format_excerpts(state.get("retrieved", [])),
            summary=state.get("summary", ""),
            question=state["question"],
        )
        state["answer"] = _invoke(llm_factory(), prompt)
        return state

    return node


DEFAULT_DIRECT_PROMPTS: Mapping[str, str] = {
    "chitchat": "direct_chitchat",
    "clarify": "direct_clarify",
}


def make_direct_answer(
    llm_factory: LLMFactory = build_chat_llm,
    prompt_names: Mapping[str, str] | None = None,
    instructions: Mapping[str, str] | None = None,
    fallback_route: str = "chitchat",
) -> Node:
    """Non-retrieval answer path.

    Resolution order for each route's instruction text:
      1. `instructions[route]` if provided (inline override, useful for tests
         or one-off use cases).
      2. `get_prompt(prompt_names[route])` -> versioned, Langfuse-editable.

    Prompts ship in `prompts/direct_chitchat.md` and `prompts/direct_clarify.md`.
    Add an entry to `prompt_names` for any new route (e.g. "escalate") so it
    picks up its own prompt file.
    """
    names = dict(prompt_names) if prompt_names is not None else dict(DEFAULT_DIRECT_PROMPTS)
    inline = dict(instructions) if instructions is not None else {}

    def _instruction_for(route: str) -> str:
        if route in inline:
            return inline[route]
        prompt_name = names.get(route) or names.get(fallback_route)
        if prompt_name is None:
            return inline.get(fallback_route, "")
        try:
            return _default_prompt_render(prompt_name)
        except Exception:
            return inline.get(fallback_route, "")

    def node(state: dict) -> dict:
        route = state.get("route", fallback_route)
        instruction = _instruction_for(route)
        prompt = f"{instruction}\n\nUSER: {state['question']}\nASSISTANT:"
        state["answer"] = _invoke(llm_factory(), prompt)
        state["retrieved"] = []
        state["citations"] = []
        return state

    return node


def make_persist_turn(memory: MemoryBackend | None = None) -> Node:
    """Append the user + assistant messages to the conversation log. No LLM call."""
    backend = memory or default_memory_backend()

    def node(state: dict) -> dict:
        session_id = state["session_id"]
        backend.append_turn(session_id, "user", state["question"])
        backend.append_turn(session_id, "assistant", state.get("answer", ""))
        return state

    return node


def make_update_summary(
    memory: MemoryBackend | None = None,
    llm_factory: LLMFactory = build_chat_llm,
    prompt_name: str = "summarizer",
) -> Node:
    """Refresh the running session summary via an LLM call. Best-effort: any
    failure is swallowed so a flaky summarizer can't break the turn."""
    backend = memory or default_memory_backend()

    def node(state: dict) -> dict:
        try:
            prompt = _default_prompt_render(
                prompt_name,
                existing_summary=state.get("summary", ""),
                user_message=state["question"],
                assistant_message=state.get("answer", ""),
            )
            new_summary = _invoke(llm_factory(), prompt)
            if new_summary:
                backend.save_summary(state["session_id"], new_summary)
        except Exception:
            pass
        return state

    return node


def make_update_memory(
    memory: MemoryBackend | None = None,
    llm_factory: LLMFactory = build_chat_llm,
    summarizer_prompt_name: str = "summarizer",
) -> Node:
    """Convenience wrapper: persist the turn and update the summary in one node.

    Prefer `make_persist_turn` + `make_update_summary` as separate graph nodes
    when you want to e.g. skip summarization every N turns, or persist to a
    different backend than the summary store.
    """
    persist = make_persist_turn(memory)
    summarize = make_update_summary(memory, llm_factory, summarizer_prompt_name)

    def node(state: dict) -> dict:
        return summarize(persist(state))

    return node
