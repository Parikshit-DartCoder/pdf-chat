"""CLI: parse PDFs, chunk, embed, upsert into Qdrant.

Usage:
    rag-ingest --pdf-dir /app/data/pdfs
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from ..config.settings import get_settings
from ..ingestion.chunker import chunk_pages
from ..ingestion.parser import parse_pdf
from ..ingestion.vector_store import upsert_chunks

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


@app.command()
def ingest(
    pdf_dir: Path = typer.Option(None, "--pdf-dir", help="Directory of PDFs to ingest."),
    chunk_size: int = typer.Option(None, "--chunk-size"),
    chunk_overlap: int = typer.Option(None, "--chunk-overlap"),
) -> None:
    s = get_settings()
    pdf_dir = pdf_dir or Path(s.pdf_input_dir)
    chunk_size = chunk_size or s.chunk_size
    chunk_overlap = chunk_overlap or s.chunk_overlap

    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]No PDFs found under {pdf_dir}[/]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Ingesting {len(pdfs)} PDF(s) from {pdf_dir}[/]")
    total_pages = 0
    total_chunks = 0
    total_written = 0

    failed: list[tuple[str, str]] = []
    with Progress() as progress:
        task = progress.add_task("Parsing and embedding", total=len(pdfs))
        for pdf in pdfs:
            try:
                pages = parse_pdf(pdf)
                chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                written = upsert_chunks(chunks)
                total_pages += len(pages)
                total_chunks += len(chunks)
                total_written += written
                console.print(f"  - {pdf.name}: {len(pages)} pages, {len(chunks)} chunks, {written} upserted")
            except Exception as e:
                failed.append((pdf.name, str(e)))
                console.print(f"  [red]failed[/] {pdf.name}: {e}")
            progress.advance(task)

    if failed:
        console.print(f"\n[yellow]Skipped {len(failed)} file(s):[/]")
        for name, err in failed:
            console.print(f"  - {name}: {err}")

    console.print(
        f"[green]Done.[/] pages={total_pages} chunks={total_chunks} upserted={total_written} "
        f"collection={s.qdrant_collection}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
