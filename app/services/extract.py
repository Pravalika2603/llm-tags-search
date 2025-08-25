import hashlib, os, fitz, pytesseract, io, pandas as pd, re
from PIL import Image
from docx import Document as Docx
from langdetect import detect
from typing import TypedDict

class Extracted(TypedDict):
    text: str
    structure: list[dict]   # [{"heading":"...", "start":int}, ...]
    pages: list[str]
    lang: str
    ocr_confidence: float | None

def _lang(text: str) -> str:
    try:
        return detect(text[:5000]) if text.strip() else "en"
    except:
        return "en"

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def from_pdf(path: str) -> Extracted:
    doc = fitz.open(path)
    pages, total_ocr = [], []
    structure = []
    for i, page in enumerate(doc):
        txt = page.get_text("text")
        if len(txt.strip()) < 30:
            # OCR fallback
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_txt = pytesseract.image_to_string(img)
            conf = 0.7 if len(ocr_txt.strip()) > 30 else 0.0
            txt = ocr_txt
            total_ocr.append(conf)
        # crude heading heuristics
        for line in txt.splitlines()[:5]:
            if re.match(r"^\s*[A-Z0-9].{0,80}$", line):
                structure.append({"heading": line.strip()[:120], "page": i+1})
                break
        pages.append(txt)
    full = "\n\n".join(pages)
    return {
        "text": full,
        "structure": structure,
        "pages": pages,
        "lang": _lang(full),
        "ocr_confidence": sum(total_ocr)/len(total_ocr) if total_ocr else None
    }

def from_docx(path: str) -> Extracted:
    d = Docx(path)
    lines, structure = [], []
    for p in d.paragraphs:
        if p.style.name.lower().startswith("heading"):
            structure.append({"heading": p.text.strip()[:120]})
            lines.append(f"## {p.text.strip()}")
        else:
            lines.append(p.text)
    text = "\n".join(lines)
    return {"text": text, "structure": structure, "pages": [], "lang": _lang(text), "ocr_confidence": None}

def from_table(path: str) -> Extracted:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    # Create a text “view” (keep the structured file in storage elsewhere if needed)
    head = ", ".join(map(str, df.columns.tolist()))
    rows = []
    for i, row in df.iterrows():
        kv = "; ".join([f"{k}: {row[k]}" for k in df.columns[:10]])  # cap columns for sanity
        rows.append(f"Row {i+1} — {kv}")
        if i > 2000: break
    text = f"TABLE COLUMNS: {head}\n" + "\n".join(rows)
    return {"text": text, "structure": [{"heading": os.path.basename(path)}], "pages": [], "lang": _lang(text), "ocr_confidence": None}

def from_txt(path: str) -> Extracted:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return {"text": text, "structure": [], "pages": [], "lang": _lang(text), "ocr_confidence": None}

def extract_any(path: str) -> Extracted:
    p = path.lower()
    if p.endswith(".pdf"):   return from_pdf(path)
    if p.endswith(".docx"):  return from_docx(path)
    if p.endswith(".csv") or p.endswith(".xlsx"): return from_table(path)
    return from_txt(path)
