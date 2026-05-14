from rag_exp.ingestion.chunker import chunk_pages
from rag_exp.ingestion.parser import ParsedPage


def test_chunks_split_long_pages():
    page = ParsedPage(source_path="x.pdf", page_number=1, text="word " * 800)
    chunks = chunk_pages([page], chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert c.source_path == "x.pdf"
        assert c.page_number == 1


def test_detects_arabic_language():
    page = ParsedPage(source_path="ar.pdf", page_number=2, text="هذا نص باللغة العربية يحتوي على عدة كلمات للاختبار " * 10)
    chunks = chunk_pages([page], chunk_size=500, chunk_overlap=50)
    assert chunks
    assert all(c.language == "ar" for c in chunks)
