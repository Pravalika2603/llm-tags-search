from sentence_transformers import SentenceTransformer
from typing import List
from ..config import settings

_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model

def embed_texts(texts: List[str]) -> list[list[float]]:
    model = get_embedder()
    # E5 instruct: prepend "query: " for queries, "passage: " for documents
    return model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True).tolist()

def embed_query(q: str) -> list[float]:
    model = get_embedder()
    return model.encode([f"query: {q}"], normalize_embeddings=True)[0].tolist()
