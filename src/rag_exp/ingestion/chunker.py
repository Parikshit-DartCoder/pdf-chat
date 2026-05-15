"""Text-cleaning + semantic chunking, AR/EN-aware.

Two-stage:
1. clean_text() strips repeated PDF page furniture (running headers, bare page
   numbers, roman numerals, rule lines, illustration credits) that otherwise
   dominate chunks of formatted/illustrated PDFs and tank retrieval precision.
2. chunk_pages() prefers semantic boundaries -- "Article N", numbered/lettered
   clauses, blank lines -- before falling back to a recursive character split.
   Smaller default chunk size keeps each chunk topically tight.
"""
from __future__ import annotations

import re
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

# Lines that are pure page furniture in formatted/illustrated PDFs.
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*$"),                                  # blank
    re.compile(r"^\s*\d{1,4}\s*$"),                        # bare page number
    re.compile(r"^\s*[ivxlcdm]{1,6}\s*$", re.IGNORECASE),  # roman numeral page
    re.compile(r"^\s*[-–—_=•·∗*]{2,}\s*$"),                # rule / divider line
    re.compile(r"^\s*\|.*\|\s*$"),                          # | running header |
    # 1-3 word ALL-CAPS line with no sentence punctuation -> running header
    # fragment (e.g. "UNITED", "NATIONS", "PREAMBLE"). Body text isn't all-caps.
    re.compile(r"^\s*[A-Z][A-Z]{1,14}(?:\s+[A-Z]{2,15}){0,2}\s*$"),
    re.compile(r"^\s*illustrations?\s+by\b.*$", re.IGNORECASE),
    re.compile(r"^\s*all rights reserved\b.*$", re.IGNORECASE),
    re.compile(r"^\s*©.*$"),                                # copyright line
    re.compile(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE),
]

# Semantic split points: keep articles / numbered clauses intact.
_SEMANTIC_SPLIT = re.compile(
    r"(?=^\s*(?:Article\s+\d+|المادة\s+\S+|\d+\.\s|\([a-z0-9]\)\s))",
    re.IGNORECASE | re.MULTILINE,
)


def _detect_language(text: str) -> str:
    """Cheap heuristic — counts Arabic-script characters."""
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    return "ar" if arabic > max(20, len(text) * 0.2) else "en"


def clean_text(text: str) -> str:
    """Drop page-furniture lines and collapse the whitespace they leave behind."""
    kept: list[str] = []
    for line in text.splitlines():
        if any(p.match(line) for p in _BOILERPLATE_PATTERNS):
            continue
        kept.append(line.rstrip())
    cleaned = "\n".join(kept)
    # Collapse 3+ newlines to a paragraph break; trim trailing spaces.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _semantic_segments(text: str) -> list[str]:
    """Split on article/clause boundaries; keep each segment whole when small."""
    parts = [p.strip() for p in _SEMANTIC_SPLIT.split(text) if p and p.strip()]
    return parts or ([text] if text.strip() else [])


def chunk_pages(
    pages: Iterable[ParsedPage],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_AR_EN_SEPARATORS,
        length_function=len,
    )

    out: list[Chunk] = []
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue

        idx = 0
        for segment in _semantic_segments(cleaned):
            # A semantic segment that already fits stays as one chunk so an
            # "Article N ..." block isn't sliced mid-sentence.
            pieces = (
                [segment]
                if len(segment) <= chunk_size
                else splitter.split_text(segment)
            )
            for piece in pieces:
                piece = piece.strip()
                if len(piece) < 20:  # drop slivers left by cleaning/splitting
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
                idx += 1
    return out
