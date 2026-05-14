"""RAGAs evaluation harness. Reads a JSONL file of {question, ground_truth},
runs the live agent on each, and reports faithfulness / answer relevancy /
context precision / context recall."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings

from ..agent.graph import run_agent
from ..config.settings import get_settings


@dataclass
class EvalCase:
    question: str
    ground_truth: str


def load_cases(path: Path) -> list[EvalCase]:
    out: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.append(EvalCase(question=row["question"], ground_truth=row["ground_truth"]))
    return out


def _agent_answers(cases: Iterable[EvalCase], session_prefix: str = "eval") -> list[dict]:
    rows: list[dict] = []
    for i, case in enumerate(cases):
        result = run_agent(case.question, session_id=f"{session_prefix}-{i}")
        contexts = []
        for c in result.citations:
            snippet = c.get("snippet", "")
            contexts.append(snippet)
        rows.append(
            {
                "question": case.question,
                "answer": result.answer,
                "contexts": contexts or ["(no context retrieved)"],
                "ground_truth": case.ground_truth,
            }
        )
    return rows


def run_evaluation(cases_path: Path) -> dict:
    cases = load_cases(cases_path)
    if not cases:
        raise ValueError(f"no eval cases found in {cases_path}")

    rows = _agent_answers(cases)
    dataset = Dataset.from_list(rows)

    s = get_settings()
    # Judge LLM gets a generous timeout: RAGAs feeds it long context+answer
    # prompts and a hot GPU can still take 30-60s per call.
    judge_llm = ChatOllama(
        model=s.llm_model,
        base_url=s.ollama_base_url,
        temperature=0.0,
        timeout=600.0,
        num_ctx=8192,
    )
    judge_embeddings = OllamaEmbeddings(model=s.embedding_model, base_url=s.ollama_base_url)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        # Cap concurrency so Ollama isn't slammed by 24 parallel jobs (which
        # caused the Job[*] TimeoutErrors). Sequential is slower but reliable.
        run_config=RunConfig(max_workers=2, timeout=600),
    )
    return {"scores": result, "rows": rows}
