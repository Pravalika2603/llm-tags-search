from pydantic import BaseModel
from typing import List, Dict, Optional

class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    tags: List[str]
    topics: List[str]
    sensitivity: str
    confidence: float

class SearchFilters(BaseModel):
    tags: Optional[List[str]] = None
    sensitivity: Optional[str] = None
    doc_type: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query: str
    k: int = 8
    filters: Optional[SearchFilters] = None
    return_answer: bool = True

class SearchHit(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    heading: Optional[str]
    page: Optional[int]
    text: str
    score: float
    tags: List[str] = []

class SearchResponse(BaseModel):
    hits: List[SearchHit]
    answer: Optional[str] = None
    citations: Optional[List[str]] = None
    suggested_facets: Optional[Dict[str, List[str]]] = None
