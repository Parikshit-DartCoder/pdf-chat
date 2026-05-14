"""Recursive character splitter, AR/EN-aware: respects Arabic punctuation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .parser import ParsedPage


@dataclass
class Chunk:
    text: str
    source_path: str
    page_number: int
    chunk_index: int  # within (source, page)
    language: str | None = None


_AR_EN_SEPARATORS = [
    "\n\n", "\n",
    "۔", "؟", "؛", "،",  # Arabic punctuation
    ".", "?", "!", ";", ",",
    " ", "",
]


def _detect_language(text: str) -> str:
    """Cheap heuristic — counts Arabic-script characters."""
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    return "ar" if arabic > max(20, len(text) * 0.2) else "en"


def chunk_pages(
    pages: Iterable[ParsedPage],
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_AR_EN_SEPARATORS,
        length_function=len,
    )

    out: list[Chunk] = []
    for page in pages:
        pieces = splitter.split_text(page.text)
        for idx, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            out.append(
                Chunk(
                    text=piece,
                    source_path=page.source_path,
                    page_number=page.page_number,
                    chunk_index=idx,
                    language=_detect_language(piece),
                )
            )
    return out
