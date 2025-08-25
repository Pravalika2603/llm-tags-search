from openai import OpenAI
from typing import List, Dict
from ..config import settings

print("Using OpenAI API key:", settings.OPENAI_API_KEY)
client = OpenAI(
    api_key=settings.OPENAI_API_KEY
)



ANSWER_SYS = (
"Answer only from the provided context. If unsure, say you don't know. "
"Always cite sources as [doc_id#chunk_idx]."
)

def build_context(hits: List[Dict], max_chars=8000) -> str:
    buf, used = [], 0
    for h in hits:
        piece = f"[{h['doc_id']}#{h['chunk_idx']}] {h['text']}\n"
        if used + len(piece) > max_chars:
            break
        buf.append(piece)
        used += len(piece)
    return "\n".join(buf)

def answer(query: str, hits: List[Dict]) -> tuple[str, list[str]]:
    ctx = build_context(hits)
    msgs = [
        {"role":"system","content":ANSWER_SYS},
        {"role":"user","content": f"Question: {query}\n\nContext:\n{ctx}"}
    ]
    # TODO: modify this model to use a more suitable one for answering- for now it is gpt-4o-mini or we will moving llama-3.1
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        temperature=0.0
    )
    text = resp.choices[0].message.content.strip()
    citations = [f"{h['doc_id']}#{h['chunk_idx']}" for h in hits[:5]]
    return text, citations
