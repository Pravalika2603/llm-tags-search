#!/usr/bin/env bash
python - <<'PY'
from sentence_transformers import SentenceTransformer
for m in ["intfloat/e5-large-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2"]:
    print("Downloading", m); SentenceTransformer(m)
print("Done.")
PY
