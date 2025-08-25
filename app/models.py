import uuid, datetime as dt
from sqlalchemy import Column, String, Text, JSON, TIMESTAMP, Integer, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from pgvector.sqlalchemy import Vector
from .db import Base

class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    source_path = Column(Text, nullable=False)
    doc_type = Column(String(16), nullable=False)
    author = Column(Text)
    lang = Column(String(8))
    sensitivity = Column(String(32), nullable=False, default="Internal")
    created_at = Column(TIMESTAMP(timezone=True), default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=dt.datetime.utcnow, nullable=False)
    tags = Column(JSON, default=list)
    entities = Column(JSON, default=dict)  # {"ORG":[], "PERSON":[], "PRODUCT":[]}
    topics = Column(JSON, default=list)
    summary = Column(Text)
    version = Column(String(16))
    confidence = Column(Float, default=0.0)
    ocr_confidence = Column(Float)
    content_hash = Column(Text, unique=True)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_idx = Column(Integer, nullable=False)
    heading = Column(Text)
    page = Column(Integer)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1024))  # keep in sync with settings.EMBEDDING_DIM
    tsv = Column(TSVECTOR)
    created_at = Column(TIMESTAMP(timezone=True), default=dt.datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="chunks")
