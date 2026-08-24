from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [np.asarray(v, dtype=np.float32) for v in vectors]


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
