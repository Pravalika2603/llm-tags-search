import json, re, time
from typing import Dict, List
from openai import OpenAI
from ..config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

TAG_GLOSSARY = {
  "doc_type": ["Policy","SOP","Contract","Invoice","Spec","Email","Report","Minutes","Playbook","Other"],
  "domain":   ["Finance","Healthcare","HR","Legal","IT","Sales","Support","Ops","Other"],
  "sensitivity": ["Public","Internal","Confidential"]
}

PROMPT = """You are a precise document tagger. Return STRICT JSON:
{{
 "doc_type": "...",
 "domain": "...",
 "topics": ["..."],
 "sensitivity": "...",
 "confidence": 0.0
}}
Constraints:
- doc_type ∈ {doc_types}
- domain ∈ {domains}
- sensitivity ∈ {sens}
Input:
Title: {title}
Excerpt:
\"\"\"{excerpt}\"\"\""""

def _extract_json(s: str) -> dict:
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def tag_document(title: str, excerpt: str) -> Dict:
    msg = PROMPT.format(
        doc_types=TAG_GLOSSARY["doc_type"],
        domains=TAG_GLOSSARY["domain"],
        sens=TAG_GLOSSARY["sensitivity"],
        title=title[:200],
        excerpt=excerpt[:8000]
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Return only valid JSON."},
                  {"role": "user", "content": msg}],
        temperature=0.0
    )
    data = _extract_json(resp.choices[0].message.content or "")
    # safety defaults
    data.setdefault("doc_type", "Other")
    data.setdefault("domain", "Other")
    data.setdefault("topics", [])
    data.setdefault("sensitivity", settings.DEFAULT_SENSITIVITY)
    data.setdefault("confidence", 0.5)
    return data
