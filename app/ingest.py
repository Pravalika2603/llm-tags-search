import os, datetime as dt, re, uuid
from sqlalchemy import text as sqltext
from sqlalchemy.orm import Session
from rapidfuzz import fuzz
from .db import get_session
from .models import Document, Chunk
from .services.extract import extract_any
from .services.chunker import chunk_text
from .services.embedder import embed_texts
from .services.tagger import tag_document
from .config import settings

SIMPLE_PII_RE = re.compile(r"(\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b|\b\d{10}\b|\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b)", re.I)

def guess_title(path:str, text:str) -> str:
    base = os.path.basename(path)
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    return first_line[:120] if len(first_line) >= 6 else base

def guess_doctype(path:str, text:str) -> str:
    if path.endswith(".pdf"): return "pdf"
    if path.endswith(".docx"): return "docx"
    if path.endswith(".csv") or path.endswith(".xlsx"): return "table"
    return "txt"

def rule_sensitivity(text:str) -> str:
    if "confidential" in text.lower() or SIMPLE_PII_RE.search(text):
        return "Confidential"
    return settings.DEFAULT_SENSITIVITY

def tsvectorize(db: Session, chunk_id):
    # Compute tsvector in SQL to use dictionaries/stemmers
    db.execute(sqltext("UPDATE chunks SET tsv = to_tsvector('english', left(text, 50000)) WHERE id = :cid"), {"cid": str(chunk_id)})

def ingest_file(db: Session, path: str) -> dict:
    ext = extract_any(path)
    title = guess_title(path, ext["text"])
    doc_type = guess_doctype(path, ext["text"])
    sensitivity = rule_sensitivity(ext["text"])

    # de-dup by content hash
    import hashlib
    h = hashlib.sha256(ext["text"].encode("utf-8")).hexdigest()
    existing = db.query(Document).filter(Document.content_hash == h).first()
    if existing:
        return {"doc_id": str(existing.id), "chunks": len(existing.chunks), "skipped": True}

    doc = Document(
        title=title,
        source_path=os.path.abspath(path),
        doc_type=doc_type,
        author=None,
        lang=ext["lang"],
        sensitivity=sensitivity,
        created_at=dt.datetime.utcnow(),
        updated_at=dt.datetime.utcnow(),
        ocr_confidence=ext["ocr_confidence"],
        content_hash=h
    )
    db.add(doc); db.flush()

    # Tag on abstract (first ~8k chars)
    tag = tag_document(title, ext["text"][:8000])
    doc.tags = [f"Domain/{tag['domain']}", f"DocType/{tag['doc_type']}"]
    doc.topics = tag.get("topics", [])
    doc.sensitivity = tag.get("sensitivity", sensitivity)
    doc.confidence = float(tag.get("confidence", 0.5))
    db.commit()

    # Chunk & embed
    chunks = chunk_text(ext["text"], ext["structure"], target_tokens=800, overlap=80)
    embeddings = embed_texts([c["text"] for c in chunks])
    for i, (c, emb) in enumerate(zip(chunks, embeddings)):
        ch = Chunk(
            doc_id=doc.id,
            chunk_idx=i,
            heading=c.get("heading"),
            page=c.get("page"),
            text=c["text"],
            embedding=emb
        )
        db.add(ch); db.flush()
        tsvectorize(db, ch.id)
    db.commit()
    return {"doc_id": str(doc.id), "chunks": len(chunks), "tags": doc.tags, "topics": doc.topics, "sensitivity": doc.sensitivity, "confidence": doc.confidence}
