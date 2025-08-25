from sentence_transformers import CrossEncoder
from typing import List, Dict
from ..config import settings

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker

def rerank(query: str, candidates: List[Dict], top_k: int = 8) -> List[Dict]:
    model = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs).tolist()
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
