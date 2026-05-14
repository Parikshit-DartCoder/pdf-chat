"""Download a small set of public-domain EN+AR PDFs into data/pdfs/ so you can
smoke-test the full pipeline without hunting for content.

Sources are official UN/UNESCO documents (multilingual, freely redistributable).

Usage:
    python scripts/fetch_sample_pdfs.py
    # or inside the container:
    docker compose exec app python /app/scripts/fetch_sample_pdfs.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import Request, urlopen


# (filename, url) pairs. Picked to total ~80 pages, balanced EN+AR.
# All public-domain UN documents.
SAMPLES: list[tuple[str, str]] = [
    # Universal Declaration of Human Rights, English (~40 pages illustrated)
    ("udhr_en.pdf",
     "https://www.un.org/en/udhrbook/pdf/udhr_booklet_en_web.pdf"),
    # Universal Declaration of Human Rights, Arabic (~40 pages illustrated)
    ("udhr_ar.pdf",
     "https://www.un.org/ar/udhrbook/pdf/udhr_booklet_ar_web.pdf"),
]


def _is_valid_pdf(path: Path) -> bool:
    """Cheap-but-effective PDF sanity check: header magic and EOF marker.
    Catches truncated downloads and HTML-as-PDF mistakes."""
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as f:
        head = f.read(8)
        if not head.startswith(b"%PDF-"):
            return False
        f.seek(-1024, 2)   # last 1 KB
        tail = f.read()
        return b"%%EOF" in tail


def download(url: str, dest: Path) -> int:
    req = Request(url, headers={"User-Agent": "rag_exp/0.1 (smoke-test fetcher)"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(req, timeout=120) as r, tmp.open("wb") as f:
        chunk = r.read(64 * 1024)
        total = 0
        while chunk:
            f.write(chunk)
            total += len(chunk)
            chunk = r.read(64 * 1024)
    # Only move into place once the write completes successfully.
    tmp.replace(dest)
    return total


def main() -> None:
    target = Path("/app/data/pdfs") if Path("/app/data/pdfs").exists() else Path("data/pdfs")
    target.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {target.resolve()}")

    ok = 0
    for name, url in SAMPLES:
        dest = target / name
        if _is_valid_pdf(dest):
            print(f"  skip  {name} (valid, {dest.stat().st_size:,} bytes)")
            ok += 1
            continue
        if dest.exists():
            print(f"  redo  {name} (existing file is corrupt or incomplete)")
            dest.unlink(missing_ok=True)
        try:
            n = download(url, dest)
            if not _is_valid_pdf(dest):
                print(f"  fail  {name}: downloaded bytes are not a valid PDF", file=sys.stderr)
                dest.unlink(missing_ok=True)
                continue
            print(f"  ok    {name} ({n:,} bytes)")
            ok += 1
        except Exception as e:
            print(f"  fail  {name}: {e}", file=sys.stderr)

    print(f"\n{ok}/{len(SAMPLES)} valid. Next:")
    print("  docker compose exec api rag-ingest")


if __name__ == "__main__":
    main()
