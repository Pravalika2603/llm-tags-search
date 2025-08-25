from fastapi import FastAPI, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import text as sqltext
from typing import List, Optional
import os, tempfile, uuid

from .db import get_session
from .models import Document, Chunk
from .schema import SearchRequest, SearchResponse, SearchHit, IngestResponse
from .services.embedder import embed_query
from .services.reranker import rerank
from .services.answerer import answer
from .ingest import ingest_file

app = FastAPI(title="LLM Tagging + Semantic Search")

@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(
    file: UploadFile = File(...),
    db: Session = Depends(get_session),
):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, file.filename)
        with open(path, "wb") as f:
            f.write(file.file.read())
        res = ingest_file(db, path)
    return res

def _filter_sql(filters) -> tuple[str, dict]:
    where = []
    params = {}
    if filters:
        if filters.sensitivity:
            where.append("d.sensitivity = :sens"); params["sens"] = filters.sensitivity
        if filters.tags:
            where.append("d.tags ?| array[:taglist]"); params["taglist"] = filters.tags
        if filters.doc_type:
            where.append("d.doc_type = ANY(:dts)"); params["dts"] = filters.doc_type
    sql = (" AND ".join(where)) if where else "TRUE"
    return sql, params

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, db: Session = Depends(get_session)):
    qvec = embed_query(req.query)

    where_sql, params = _filter_sql(req.filters)
    # Vector top-N (cosine) and keyword top-M
    params.update({"qvec": qvec, "k": req.k * 4, "m": req.k * 2, "query": req.query})
    sql = f"""
    WITH vec AS (
      SELECT c.id as chunk_id, c.doc_id, c.chunk_idx, c.text, c.heading, c.page,
             1 - (c.embedding <=> :qvec::vector) AS vec_score
      FROM chunks c
      JOIN documents d ON d.id = c.doc_id
      WHERE {where_sql}
      ORDER BY c.embedding <=> :qvec::vector
      LIMIT :k
    ),
    kw AS (
      SELECT c.id as chunk_id, c.doc_id, c.chunk_idx, c.text, c.heading, c.page,
             ts_rank_cd(c.tsv, plainto_tsquery('english', :query)) AS kw_score
      FROM chunks c
      JOIN documents d ON d.id = c.doc_id
      WHERE {where_sql} AND c.tsv @@ plainto_tsquery('english', :query)
      ORDER BY kw_score DESC
      LIMIT :m
    ),
    merged AS (
      SELECT * FROM vec
      UNION
      SELECT * FROM kw
    )
    SELECT m.*, d.title, d.tags
    FROM merged m JOIN documents d ON d.id = m.doc_id
    """
    rows = db.execute(sqltext(sql), params).mappings().all()
    candidates = []
    for r in rows:
        score = float((r.get("vec_score") or 0) + (r.get("kw_score") or 0))
        candidates.append({
            "doc_id": str(r["doc_id"]),
            "chunk_id": str(r["chunk_id"]),
            "chunk_idx": int(r["chunk_idx"]),
            "title": r["title"],
            "heading": r["heading"],
            "page": r["page"],
            "text": r["text"],
            "score": score,
            "tags": r["tags"] or []
        })

    # Optional cross-encoder rerank to top-k
    ranked = rerank(req.query, candidates, top_k=req.k) if len(candidates) > req.k else sorted(candidates, key=lambda x: x["score"], reverse=True)[:req.k]

    hits = [SearchHit(
        doc_id=h["doc_id"], chunk_id=h["chunk_id"], title=h["title"], heading=h["heading"], page=h["page"],
        text=h["text"][:1200], score=h.get("rerank_score", h["score"]), tags=h["tags"]
    ) for h in ranked]

    out = {"hits": hits, "answer": None, "citations": None, "suggested_facets": {
        "DocType": list({t.split("/",1)[1] for h in ranked for t in (h["tags"] or []) if t.startswith("DocType/")}),
        "Domain": list({t.split("/",1)[1] for h in ranked for t in (h["tags"] or []) if t.startswith("Domain/")})
    }}
    if req.return_answer:
        ans, cites = answer(req.query, ranked)
        out["answer"] = ans
        out["citations"] = cites
    return out
