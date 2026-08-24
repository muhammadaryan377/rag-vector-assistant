from app.ingestion import chunk_text


def test_chunk_text_returns_multiple_chunks():
    text = "AI systems retrieve useful context. " * 100
    chunks = chunk_text(text, chunk_size=200, overlap=30)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_empty_input():
    assert chunk_text("   ") == []
