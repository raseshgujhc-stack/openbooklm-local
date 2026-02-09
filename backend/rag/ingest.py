# rag/ingest.py

import uuid
import json
import hashlib
import re
from datetime import datetime, date

from rag.chunking import split_into_chunks
from rag.embedder import embed_texts
from rag.vector_store import save_vectors
from rag.llm import llm
from db import get_repo


# ============================================================
# UTILS
# ============================================================

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_date(raw):
    if not raw:
        return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except Exception:
            pass
    return None


# ============================================================
# DETERMINISTIC EXTRACTORS (COURT-AGNOSTIC)
# ============================================================

def extract_basic(text):
    return {
        "word_count": len(text.split()),
        "language": "en",
        "source_type": "pdf",
        "has_citations": detect_citations(text),
        "has_tables": "TABLE" in text.upper(),
        "has_annexures": "ANNEXURE" in text.upper(),
    }


def detect_citations(text):
    patterns = [
        r"\bSCC\b",
        r"\bAIR\b",
        r"\bCriLJ\b",
        r"\bSCR\b",
        r"\bGLH\b",
        r"\bGuj\s*LR\b",
        r"\(\d{4}\)\s*\d+\s*SCC",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def extract_document_role(text):
    h = text[:1500].upper()
    if "JUDGMENT" in h or "ORDER" in h:
        return "Judicial"
    if "CIRCULAR" in h or "OFFICE MEMORANDUM" in h:
        return "Circular"
    return "Administrative"


def extract_court(text):
    header = text[:3000].upper()

    # Supreme Court
    if "SUPREME COURT OF INDIA" in header:
        return {
            "court_name": "Supreme Court of India",
            "court_level": "Supreme Court",
            "jurisdiction": {"country": "India"},
        }

    # High Courts (robust stop conditions)
    m = re.search(
        r"HIGH COURT OF\s+([A-Z ]+?)(?:\s+AT|\n|,|$)",
        header
    )
    if m:
        state = m.group(1).strip().title()
        return {
            "court_name": f"High Court of {state}",
            "court_level": "High Court",
            "jurisdiction": {"state": state},
        }

    # Tribunals
    if "TRIBUNAL" in header:
        return {
            "court_name": None,
            "court_level": "Tribunal",
            "jurisdiction": None,
        }

    return {}



def extract_case_number(text):
    m = re.search(r"\b[A-Z/]*\d+/\d{4}\b", text[:2000])
    return m.group(0) if m else None


def extract_order_date(text):
    m = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", text[:2000])
    return normalize_date(m.group(0)) if m else None


def extract_judges_and_bench(text):
    header = text[:5000]

    # Step 1: Find CORAM / BEFORE block
    m = re.search(
        r"(CORAM|BEFORE|PRESENT)\s*[:\-]?\s*(.*)",
        header,
        re.IGNORECASE
    )
    if not m:
        return [], []

    block = m.group(2)

    # Step 2: Split into logical lines (OCR-safe)
    lines = re.split(r"[\n\r]+", block)

    judges = []
    buffer = ""

    for line in lines:
        clean = line.strip()
        if not clean:
            continue

        upper = clean.upper()

        # HARD STOP: not part of names
        if any(k in upper for k in ["DATE", "DATED", "ORAL", "JUDGMENT", "ORDER"]):
            break

        # If line starts a judge
        if "JUSTICE" in upper:
            if buffer:
                judges.append(buffer.strip())
            buffer = clean
        else:
            # Continuation of name (e.g. MOOL CHAND TYAGI)
            if buffer:
                buffer += " " + clean

    if buffer:
        judges.append(buffer.strip())

    # Step 3: Final validation & cleanup
    final = []
    for j in judges:
        # Must contain JUSTICE
        if "JUSTICE" not in j.upper():
            continue
        # Must not contain digits
        if re.search(r"\d", j):
            continue
        final.append(j.title())

    return final, final




def infer_document_type_and_stage(text):
    header = text[:2000].upper()
    if "JUDGMENT" in header:
        return "Judgment", "Final"
    if "ORDER" in header:
        return "Order", "Interim"
    return None, None


def extract_decision_status(text):
    tail = text[-3000:].lower()
    patterns = {
        "Allowed": ["allowed"],
        "Dismissed": ["dismissed"],
        "Disposed": ["disposed of"],
        "Partly Allowed": ["partly allowed"],
    }
    for status, keys in patterns.items():
        for k in keys:
            if k in tail:
                return status
    return None


def extract_acts(text):
    acts = set()
    for m in re.finditer(r"([A-Z][A-Za-z\s]+ Act)", text):
        acts.add(m.group(1).strip())
    return [{"act": a, "sections": []} for a in acts]

def normalize_and_dedup_acts(acts):
    seen = {}
    for a in acts:
        raw_name = a.get("act")
        canon = normalize_act_name(raw_name)
        if not canon:
            continue

        # use canonical name as key
        if canon not in seen:
            seen[canon] = {
                "act": canon,
                "sections": a.get("sections", [])
            }

    return list(seen.values())



def document_length_class(word_count):
    if word_count < 2000:
        return "Short"
    if word_count < 8000:
        return "Medium"
    return "Long"


# ============================================================
# LLM EXTRACTION (SEMANTIC ONLY)
# ============================================================

def extract_llm_metadata(text):
    prompt = f"""
You are an Indian judicial document metadata extractor.

RULES:
- Extract only if reasonably inferable
- Do NOT hallucinate facts
- Return VALID JSON ONLY

JSON:
{{
  "case_type": null,
  "parties": null,
  "filing_side": null,
  "primary_topics": [],
  "secondary_topics": [],
  "keywords": [],
  "referenced_laws": [],
  "cited_cases": [],
  "cited_courts": [],
  "cited_acts": [],
  "final_directions": null,
  "document_about": null
}}

TEXT:
{text[:6000] + text[-4000:]}
"""
    res = llm(prompt, temperature=0.0, max_tokens=700)
    try:
        return json.loads(res["choices"][0]["text"])
    except Exception:
        return {}


# ============================================================
# CONFIDENCE
# ============================================================

def compute_confidence(meta):
    total, filled = 0, 0
    field_conf = {}

    for k, v in meta.items():
        total += 1
        ok = False
        if isinstance(v, list):
            ok = len(v) > 0
        else:
            ok = v not in (None, "", False)

        if ok:
            filled += 1
            field_conf[k] = 0.9
        else:
            field_conf[k] = 0.0

    return round(filled / total, 2), field_conf


# ============================================================
# MAIN INGEST
# ============================================================

def ingest_document(
    text: str,
    document_id: str | None = None,
    collection_id: str | None = None,
    user_id: str | None = None,
    filename: str | None = None,
):
    if not document_id:
        document_id = str(uuid.uuid4())

    # ---- Vector ingestion ----
    chunks = split_into_chunks(text)
    embeddings = embed_texts(chunks)
    save_vectors(
        notebook_id=document_id,
        vectors=[{"text": c, "embedding": e} for c, e in zip(chunks, embeddings)],
        collection_id=collection_id,
    )

    # ---- Deterministic ----
    basic = extract_basic(text)
    role = extract_document_role(text)
    court = extract_court(text)
    if "court_name" in court:
        court["court_name"] = clean_court_name(court["court_name"])

    if "jurisdiction" in court:
        court["jurisdiction"] = clean_jurisdiction(court["jurisdiction"])

    case_no = extract_case_number(text)
    order_dt = extract_order_date(text)
    judges, bench = extract_judges_and_bench(text)
    doc_type, case_stage = infer_document_type_and_stage(text)
    decision_status = extract_decision_status(text)
    raw_acts = extract_acts(text)
    acts = normalize_and_dedup_acts(raw_acts)


    # ---- LLM ----
    llm_meta = extract_llm_metadata(text)

    metadata = {
        "document_id": document_id,
        "filename": filename,
        "user_id": user_id,
        "collection_id": collection_id,

        "document_role": role,
        "domain": "Judicial" if role == "Judicial" else "General",

        **basic,
        **court,
        **llm_meta,

        "case_number": case_no,
        "order_date": order_dt,
        "judge_name": judges,
        "bench": bench,
        "document_type": doc_type,
        "case_stage": case_stage,
        "decision_status": decision_status,
        "outcome": decision_status,
        "act_names": acts,
        "document_length_class": document_length_class(basic["word_count"]),

        "file_hash": sha256(text),
        "created_at": datetime.utcnow().isoformat(),
        "ingested_at": datetime.utcnow().isoformat(),
        "intended_audience": "Mixed",
    }

    # ---- Confidence & explainability ----
    doc_conf, field_conf = compute_confidence(metadata)
    metadata["metadata_confidence"] = str(doc_conf)
    metadata["field_confidence"] = field_conf
    metadata["extraction_notes"] = {
        "court": "header regex",
        "judges": "coram/before regex",
        "acts": "regex + llm",
        "decision": "tail heuristic",
    }

    repo = get_repo()
    repo.insert_document(metadata)

    return {
        "document_id": document_id,
        "chunks": len(chunks),
        "confidence": doc_conf,
    }

def clean_court_name(name: str | None):
    if not name:
        return None

    # Remove "AT <CITY>", extra symbols, line breaks
    name = re.sub(r"\s+AT\s+.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[\n\r\+]+", " ", name)
    return name.strip()


def clean_jurisdiction(jurisdiction: dict | None):
    if not jurisdiction or not isinstance(jurisdiction, dict):
        return jurisdiction

    state = jurisdiction.get("state")
    if state:
        state = re.sub(r"\s+AT\s+.*$", "", state, flags=re.IGNORECASE)
        state = re.sub(r"[\n\r]+", " ", state)
        jurisdiction["state"] = state.strip()

    return jurisdiction


def normalize_act_name(name: str):
    if not name:
        return None

    # normalize whitespace
    name = re.sub(r"\s+", " ", name).strip()

    # common canonical mappings (expand over time)
    CANONICAL_ACTS = {
        "motor vehicle act": "Motor Vehicles Act",
        "motor vehicles act": "Motor Vehicles Act",
        "mv act": "Motor Vehicles Act",
    }

    key = name.lower()
    key = key.replace(".", "").strip()

    return CANONICAL_ACTS.get(key, name)

