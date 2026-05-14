"""CLI: run RAGAs evaluation against a JSONL of {question, ground_truth} cases."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..evaluation.ragas_eval import run_evaluation

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _aggregate(v) -> float | None:
    """Reduce a metric value to a single float. RAGAs' `_scores_dict` maps
    metric -> list[float] (per-row scores); some versions store a single
    aggregated float. Handle both, skip NaN / None."""
    import math
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        vals = [float(x) for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return (sum(vals) / len(vals)) if vals else None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except Exception:
        return None


def _scores_dict(scores) -> dict[str, float]:
    """Pull metric -> aggregated score out of a RAGAs EvaluationResult. RAGAs
    versions differ on shape; be defensive."""
    out: dict[str, float] = {}
    raw = getattr(scores, "_scores_dict", None)
    if isinstance(raw, dict):
        for k, v in raw.items():
            agg = _aggregate(v)
            if agg is not None:
                out[k] = agg
        if out:
            return out
    # Fall back: try direct access by metric name.
    for name in METRIC_NAMES:
        try:
            agg = _aggregate(scores[name])
            if agg is not None:
                out[name] = agg
        except Exception:
            pass
    return out


@app.command()
def evaluate(
    cases: Path = typer.Argument(..., exists=True, readable=True, help="JSONL eval cases."),
    out: Path = typer.Option(Path("/app/data/eval_results.json"), "--out"),
) -> None:
    result = run_evaluation(cases)
    scores = _scores_dict(result["scores"])

    table = Table(title="RAGAs scores")
    table.add_column("metric")
    table.add_column("score", justify="right")
    for name in METRIC_NAMES:
        if name in scores:
            table.add_row(name, f"{scores[name]:.3f}")
        else:
            table.add_row(name, "n/a")
    console.print(table)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"scores": scores, "rows": result["rows"]}, indent=2),
        encoding="utf-8",
    )
    console.print(f"[green]Results written to {out}[/]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
