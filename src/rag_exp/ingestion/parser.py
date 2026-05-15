"""PDF parser.

Two-stage strategy:
1. Docling (with OCR disabled) extracts embedded text + layout. Fast for
   text-native PDFs and preserves table structure.
2. For pages where Docling returns little/no text (i.e. scans), the page is
   rasterized and sent to PaddleOCR-VL via vLLM.

PaddleOCR-VL replaces EasyOCR — better Arabic quality, GPU-batched throughput,
and structured output (markdown / latex / table HTML) handled by the VL model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..config.settings import get_settings


@dataclass
class ParsedPage:
    source_path: str
    page_number: int  # 1-indexed
    text: str
    extractor: str = "unknown"   # "docling" | "paddleocr-vl" | "pypdf"
    language: str | None = None


# ---------------------------------------------------------------------------
# Per-extractor implementations
# ---------------------------------------------------------------------------

def _parse_with_docling(pdf_path: Path) -> list[ParsedPage]:
    """Docling with OCR disabled — fast text/layout extraction only."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.do_ocr = False                # OCR handled by PaddleOCR-VL downstream
    opts.do_table_structure = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    result = converter.convert(str(pdf_path))
    doc = result.document

    pages: dict[int, list[str]] = {}
    for item in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text:
            continue
        prov = getattr(item, "prov", None) or []
        for p in prov:
            page_no = getattr(p, "page_no", None) or getattr(p, "page", None)
            if page_no is None:
                continue
            pages.setdefault(int(page_no), []).append(text)

    return [
        ParsedPage(
            source_path=str(pdf_path),
            page_number=pn,
            text="\n".join(chunks),
            extractor="docling",
        )
        for pn, chunks in sorted(pages.items())
        if "".join(chunks).strip()
    ]


def _parse_with_pypdf(pdf_path: Path) -> list[ParsedPage]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    out: list[ParsedPage] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append(ParsedPage(
                source_path=str(pdf_path),
                page_number=i,
                text=text,
                extractor="pypdf",
            ))
    return out


def _pdf_page_count(pdf_path: Path) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def _ocr_pages_parallel(pdf_path: Path, page_numbers: list[int], max_workers: int) -> dict[int, ParsedPage]:
    """OCR multiple pages, fast AND safe.

    pypdfium2 is NOT thread-safe -- opening PdfDocument from worker threads
    corrupts pdfium's global state and makes subsequent PDFs fail to load
    ("Data format error") even though they're valid. So:
      Phase 1: rasterize every needed page SEQUENTIALLY (single-threaded pdfium).
      Phase 2: send the page images to PaddleOCR-VL CONCURRENTLY (HTTP is the
               slow part and is thread-safe).
    """
    if not page_numbers:
        return {}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .paddleocr_vl_client import ocr_page, rasterize_pdf_page

    # Phase 1: single-threaded rasterization.
    rasterized: dict[int, bytes] = {}
    for pn in page_numbers:
        try:
            rasterized[pn] = rasterize_pdf_page(str(pdf_path), pn)
        except Exception:
            continue

    # Phase 2: concurrent OCR over the in-memory PNGs (no pdfium in threads).
    def _ocr(pn: int, png: bytes) -> ParsedPage | None:
        try:
            result = ocr_page(png, page_number=pn, task="OCR")
        except Exception:
            return None
        if not result.text.strip():
            return None
        return ParsedPage(
            source_path=str(pdf_path),
            page_number=pn,
            text=result.text,
            extractor="paddleocr-vl",
        )

    results: dict[int, ParsedPage] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {pool.submit(_ocr, pn, png): pn for pn, png in rasterized.items()}
        for fut in as_completed(futures):
            page = None
            try:
                page = fut.result()
            except Exception:
                page = None
            if page is not None:
                results[page.page_number] = page
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: Path) -> list[ParsedPage]:
    """Parse a PDF page-by-page using the cheapest extractor that produces
    enough text per page.

    Strategy (changed for speed):
      1. pypdf first -- ~10-50 ms/page, no GPU, no model load.
      2. If pypdf returned little text overall (<40 chars/page average), try
         Docling for the whole document (better layout + table extraction).
      3. For any page still under `ocr_min_chars_per_page`, OCR it with
         PaddleOCR-VL.

    On a text-native corpus this skips Docling entirely (the big slow step
    on every page).
    """
    s = get_settings()
    page_count = _pdf_page_count(pdf_path)
    min_chars = s.ocr_min_chars_per_page

    # Phase 1: cheap pypdf pass.
    pypdf_pages = _parse_with_pypdf(pdf_path)
    by_page: dict[int, ParsedPage] = {p.page_number: p for p in pypdf_pages}

    # Phase 2: if pypdf looks sparse overall, try Docling for layout-aware
    # extraction. Tables, multi-column, etc.
    pypdf_chars = sum(len(p.text) for p in pypdf_pages)
    if pypdf_chars < min_chars * max(1, page_count):
        try:
            docling_pages = _parse_with_docling(pdf_path)
        except Exception:
            docling_pages = []
        # Prefer Docling output only where it actually beat pypdf in char count.
        for dp in docling_pages:
            existing = by_page.get(dp.page_number)
            if existing is None or len(dp.text) > len(existing.text):
                by_page[dp.page_number] = dp

    # Phase 3: OCR any page still below the threshold. Pages are OCR'd in
    # parallel against the PaddleOCR-VL HTTP endpoint.
    pages_needing_ocr = [
        pn for pn in range(1, page_count + 1)
        if (by_page.get(pn) is None or len(by_page[pn].text) < min_chars)
    ]
    if pages_needing_ocr:
        ocr_results = _ocr_pages_parallel(pdf_path, pages_needing_ocr, s.ocr_max_concurrency)
        by_page.update(ocr_results)

    return [by_page[k] for k in sorted(by_page)]


def parse_directory(pdf_dir: Path) -> Iterable[ParsedPage]:
    for pdf in sorted(pdf_dir.rglob("*.pdf")):
        yield from parse_pdf(pdf)
