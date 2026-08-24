from app.db import similarity_search
from app.embeddings import embed_query
from app.llm import generate_answer


def ask(question: str, top_k: int = 5) -> dict:
    query_vector = embed_query(question)
    hits = similarity_search(query_vector, top_k=top_k)

    if not hits:
        return {
            "answer": "I do not have any indexed documents to answer from yet.",
            "sources": [],
        }

    context_parts = []
    sources = []

    for i, hit in enumerate(hits, start=1):
        context_parts.append(
            f"[{i}] Source: {hit['source']} | chunk: {hit['chunk_index']}\n"
            f"{hit['content']}"
        )
        sources.append(
            {
                "citation": f"[{i}]",
                "source": hit["source"],
                "chunk_index": hit["chunk_index"],
                "score": round(hit["score"], 4),
            }
        )

    answer = generate_answer(question, "\n\n".join(context_parts))
    return {"answer": answer, "sources": sources}
