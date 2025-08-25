import re
from typing import List, Dict
import nltk
import tiktoken
nltk.download('punkt', quiet=True)

def _sentences(text: str) -> List[str]:
    # fallback to regex if punkt fails
    try:
        return nltk.sent_tokenize(text)
    except:
        return re.split(r'(?<=[.!?])\s+', text)

def chunk_text(text: str, structure: List[Dict], target_tokens=800, overlap=80, model_name="gpt-4o-mini") -> List[Dict]:
    enc = tiktoken.encoding_for_model("gpt-4o-mini") if model_name else tiktoken.get_encoding("cl100k_base")
    sents = _sentences(text)
    chunks, buff, count = [], [], 0
    i = 0
    for s in sents:
        t = len(enc.encode(s))
        if count + t > target_tokens and buff:
            chunk_text = " ".join(buff)
            chunks.append({"text": chunk_text, "heading": None, "page": None, "idx": i})
            i += 1
            # overlap by tokens (approx by sentences)
            buff = buff[-max(1, overlap // max(1, t)):]
            count = sum(len(enc.encode(x)) for x in buff)
        buff.append(s)
        count += t
    if buff:
        chunks.append({"text": " ".join(buff), "heading": None, "page": None, "idx": i})
    # Optionally inject structure headings into first sentence of a chunk
    # (kept simple here)
    return chunks
