"""LangGraph agent state machine.

Default flow:
    load_memory, classify, then one of {chitchat, clarify, retrieve}.
        retrieve path: rewrite_query, retrieve, generate_answer.
        chitchat or clarify path: direct_answer.
    Finally persist_turn, update_summary, END.

Node names that match state-schema keys are rejected by LangGraph, which is
why the classifier and answerer are exposed as `classify` and `generate_answer`
rather than `route` and `answer`. State keys for those values are unchanged.

Nodes are built by factories in `nodes.py`, so any node, its LLM, prompt,
retriever, citation builder, or memory backend, can be swapped per use case
via the `nodes` argument to `build_graph`. See `build_graph` docstring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypedDict

from langgraph.graph import StateGraph, END

from ..observability.langfuse_client import get_callback_handler
from ..retrieval.retriever import RetrievedChunk
from .nodes import (
    MemoryBackend,
    Node,
    default_memory_backend,
    make_answer,
    make_direct_answer,
    make_load_memory,
    make_persist_turn,
    make_retrieve,
    make_rewrite_query,
    make_router,
    make_update_summary,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    session_id: str
    question: str
    summary: str
    route: str            # "retrieve" | "chitchat" | "clarify" | <custom>
    search_query: str
    source_paths: list[str]   # if non-empty, retrieval is scoped to these docs
    retrieved: list[RetrievedChunk]
    answer: str
    citations: list[dict]


# ---------------------------------------------------------------------------
# Graph composition
# ---------------------------------------------------------------------------

# Names that can appear in the `nodes` override mapping.
# NOTE: must not collide with any field name in AgentState (LangGraph 0.2.50+
# rejects node names equal to state keys), which is why we use "classify" and
# "generate_answer" rather than "route" / "answer".
_NODE_NAMES = (
    "load_memory",
    "classify",
    "rewrite_query",
    "retrieve",
    "generate_answer",
    "direct_answer",
    "persist_turn",
    "update_summary",
)


def _route_branch(state: AgentState) -> str:
    return "retrieve" if state.get("route") == "retrieve" else "direct"


def build_graph(
    nodes: Mapping[str, Node] | None = None,
    memory: MemoryBackend | None = None,
    state_schema: type = AgentState,
    route_branch=_route_branch,
):
    """Compose the agent graph. Pass `nodes={name: factory(...)}` to override any node.

    Examples:

        # Different LLM for the answer node only.
        graph = build_graph(nodes={
            "answer": make_answer(llm_factory=my_gpt4),
        })

        # Domain-specific retriever.
        graph = build_graph(nodes={
            "retrieve": make_retrieve(retriever=legal_retriever),
        })

        # Add an "escalate" route.
        graph = build_graph(nodes={
            "route": make_router(
                allowed_routes=("retrieve", "chitchat", "clarify", "escalate"),
            ),
            "direct_answer": make_direct_answer(instructions={
                "chitchat": "<chitchat prompt>",
                "clarify": "<clarify prompt>",
                "escalate": "Escalate to human.",
            }),
        })
    """
    overrides = dict(nodes or {})
    unknown = set(overrides) - set(_NODE_NAMES)
    if unknown:
        raise ValueError(f"unknown node names in overrides: {sorted(unknown)}")

    mem_backend = memory or default_memory_backend()
    n = {
        "load_memory":     overrides.get("load_memory",     make_load_memory(mem_backend)),
        "classify":        overrides.get("classify",        make_router()),
        "rewrite_query":   overrides.get("rewrite_query",   make_rewrite_query()),
        "retrieve":        overrides.get("retrieve",        make_retrieve()),
        "generate_answer": overrides.get("generate_answer", make_answer()),
        "direct_answer":   overrides.get("direct_answer",   make_direct_answer()),
        "persist_turn":    overrides.get("persist_turn",    make_persist_turn(mem_backend)),
        "update_summary":  overrides.get("update_summary",  make_update_summary(mem_backend)),
    }

    g = StateGraph(state_schema)
    for name, fn in n.items():
        g.add_node(name, fn)

    g.set_entry_point("load_memory")
    g.add_edge("load_memory", "classify")
    g.add_conditional_edges(
        "classify",
        route_branch,
        {"retrieve": "rewrite_query", "direct": "direct_answer"},
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("retrieve", "generate_answer")
    g.add_edge("generate_answer", "persist_turn")
    g.add_edge("direct_answer", "persist_turn")
    g.add_edge("persist_turn", "update_summary")
    g.add_edge("update_summary", END)

    compiled = g.compile()
    mem_backend.init()
    return compiled


# Cached default graph for the Streamlit app. Custom graphs should call
# `build_graph(...)` directly and pass the result to `run_agent_on(...)`.
_default_graph = None


def get_graph():
    global _default_graph
    if _default_graph is None:
        _default_graph = build_graph()
    return _default_graph


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    route: str = "retrieve"


def run_agent_on(
    graph: Any,
    question: str,
    session_id: str,
    user_id: str | None = None,
) -> AgentResult:
    """Run any compiled graph. Use this when you've built a custom graph."""
    handler = get_callback_handler(session_id=session_id, user_id=user_id)
    config = {"callbacks": [handler]} if handler is not None else {}
    initial: AgentState = {"question": question, "session_id": session_id}
    final: AgentState = graph.invoke(initial, config=config)
    return AgentResult(
        answer=final.get("answer", ""),
        citations=final.get("citations", []),
        route=final.get("route", "retrieve"),
    )


def run_agent(question: str, session_id: str, user_id: str | None = None) -> AgentResult:
    """Convenience wrapper that uses the default graph."""
    return run_agent_on(get_graph(), question, session_id, user_id)
