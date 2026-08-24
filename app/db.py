from typing import Any
import psycopg
from pgvector.psycopg import register_vector

from app.config import get_settings


def get_connection():
    settings = get_settings()
    return psycopg.connect(settings.database_url)


def init_db() -> None:
    settings = get_settings()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id BIGSERIAL PRIMARY KEY,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR({settings.embedding_dimensions}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
                ON chunks USING hnsw (embedding vector_cosine_ops);
                """
            )
        conn.commit()


def insert_chunks(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    with get_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (source, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    (
                        row["source"],
                        row["chunk_index"],
                        row["content"],
                        row["embedding"],
                    )
                    for row in rows
                ],
            )
        conn.commit()
    return len(rows)


def similarity_search(query_embedding, top_k: int = 5) -> list[dict[str, Any]]:
    with get_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, source, chunk_index, content,
                       1 - (embedding <=> %s) AS score
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "source": row[1],
            "chunk_index": row[2],
            "content": row[3],
            "score": float(row[4]),
        }
        for row in rows
    ]
