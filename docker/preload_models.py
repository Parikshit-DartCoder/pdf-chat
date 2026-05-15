"""Run at image build time. Warms every model cache so the first chat / first
ingest has zero downloads.

Best-effort: a failure here logs a warning but does not fail the build, so a
transient HF outage at build time doesn't break the whole image."""
from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path


def _warn(stage: str, err: Exception) -> None:
    print(f"[preload] {stage} failed: {err}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)


def preload_reranker() -> None:
    from FlagEmbedding import FlagReranker  # type: ignore

    FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)


def preload_docling() -> None:
    from pypdf import PdfWriter
    from docling.document_converter import DocumentConverter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    blank = Path("/tmp/blank.pdf")
    blank.write_bytes(buf.getvalue())

    DocumentConverter().convert(str(blank))


def main() -> None:
    for name, fn in [("reranker", preload_reranker), ("docling", preload_docling)]:
        print(f"[preload] {name}", flush=True)
        try:
            fn()
            print(f"[preload] {name} done", flush=True)
        except Exception as e:
            _warn(name, e)


if __name__ == "__main__":
    main()
