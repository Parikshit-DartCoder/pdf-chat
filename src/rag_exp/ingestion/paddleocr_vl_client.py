"""Client for PaddleOCR-VL served behind vLLM (OpenAI-compatible API).

PaddleOCR-VL is a VL model: it takes a page image and a task prompt, and
returns text/markdown. Supported task prompts per the PaddleOCR docs:

    "OCR:"                  text-line localisation and recognition (default)
    "Table Recognition:"    table structure as HTML or markdown
    "Formula Recognition:"  math formulas as LaTeX
    "Chart Recognition:"    chart data as structured text

For full-page document parsing we use "OCR:". The VL model handles layout
and multi-column text implicitly via its NaViT dynamic-resolution encoder.
"""
from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from typing import Literal

import httpx

from ..config.settings import get_settings


Task = Literal["OCR", "Table Recognition", "Formula Recognition", "Chart Recognition"]


@dataclass
class OCRResult:
    text: str
    task: Task
    page_number: int
    elapsed_ms: int


def _png_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def ocr_page(
    png_bytes: bytes,
    page_number: int,
    task: Task = "OCR",
) -> OCRResult:
    """Send one page image to PaddleOCR-VL and return its markdown/text."""
    s = get_settings()
    payload = {
        "model": s.paddleocr_vl_model,
        "temperature": 0.0,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _png_to_data_url(png_bytes)}},
                    {"type": "text", "text": f"{task}:"},
                ],
            }
        ],
    }
    t0 = time.time()
    with httpx.Client(timeout=s.paddleocr_vl_timeout_s) as c:
        r = c.post(f"{s.paddleocr_vl_url}/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    text = data["choices"][0]["message"]["content"] or ""
    return OCRResult(
        text=text.strip(),
        task=task,
        page_number=page_number,
        elapsed_ms=int((time.time() - t0) * 1000),
    )


def health_check(timeout_s: int = 5) -> bool:
    """Cheap liveness probe — used by the parser to decide whether to OCR or skip."""
    s = get_settings()
    try:
        with httpx.Client(timeout=timeout_s) as c:
            r = c.get(f"{s.paddleocr_vl_url}/models")
            return r.status_code == 200
    except Exception:
        return False


def rasterize_pdf_page(pdf_path: str, page_number: int, dpi: int = 200) -> bytes:
    """Render a single PDF page to PNG bytes. page_number is 1-indexed."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page = pdf[page_number - 1]
        scale = dpi / 72  # PDF user-space unit is 1/72 inch
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        pdf.close()
