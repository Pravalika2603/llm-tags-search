CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Documents (one row per file)
CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY,
  title TEXT NOT NULL,
  source_path TEXT NOT NULL,
  doc_type TEXT NOT NULL,             -- pdf|docx|table|txt
  author TEXT,
  lang TEXT,
  sensitivity TEXT NOT NULL DEFAULT 'Internal',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
  tags JSONB DEFAULT '[]',
  entities JSONB DEFAULT '{"ORG":[],"PERSON":[],"PRODUCT":[]}',
  topics JSONB DEFAULT '[]',
  summary TEXT,
  version TEXT,
  confidence DOUBLE PRECISION DEFAULT 0.0,
  ocr_confidence DOUBLE PRECISION,
  content_hash TEXT UNIQUE            -- for de-dup
);

-- Chunks (search unit)
CREATE TABLE IF NOT EXISTS chunks (
  id UUID PRIMARY KEY,
  doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  chunk_idx INT NOT NULL,
  heading TEXT,
  page INT,
  text TEXT NOT NULL,
  embedding vector(1024),             -- match EMBEDDING_DIM
  tsv tsvector,                       -- for BM25-ish keyword
  created_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Fast keyword search
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);

-- Vector index (cosine)
-- Important: set lists appropriately to your data size (IVFFLAT needs ANALYZE after insert)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Helpful JSON indexes (optional)
CREATE INDEX IF NOT EXISTS idx_documents_tags_gin ON documents USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_documents_entities_gin ON documents USING GIN (entities);
