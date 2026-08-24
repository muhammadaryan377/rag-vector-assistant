from io import BytesIO
from pathlib import Path
from pypdf import PdfReader

from app.db import insert_chunks
from app.embeddings import embed_texts


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunk = clean[start:end]

        if end < len(clean):
            boundary = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("! "))
            if boundary > chunk_size * 0.55:
                end = start + boundary + 1
                chunk = clean[start:end]

        chunks.append(chunk.strip())

        if end >= len(clean):
            break

        start = max(end - overlap, start + 1)

    return chunks


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="ignore")

    raise ValueError("Supported file types: .pdf, .txt, .md")


def ingest_document(filename: str, content: bytes) -> dict:
    text = extract_text(filename, content)
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No readable text found in the uploaded file.")

    embeddings = embed_texts(chunks)

    rows = [
        {
            "source": filename,
            "chunk_index": index,
            "content": chunk,
            "embedding": vector,
        }
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings))
    ]

    inserted = insert_chunks(rows)
    return {"source": filename, "chunks_inserted": inserted}
