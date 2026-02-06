# rag/ingest.py

import uuid
import sqlite3
from pathlib import Path
from datetime import datetime
import hashlib
import re

from rag.chunking import split_into_chunks
from rag.embedder import embed_texts
from rag.vector_store import save_vectors
from rag.llm import llm


# ============================================================
# DATABASE PATH (SAFE, ABSOLUTE)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "notebooks.db"


# ============================================================
# BASIC (DETERMINISTIC) METADATA EXTRACTION
# ============================================================

def extract_basic_metadata(text: str) -> dict:
    return {
        "word_count": len(text.split()),
        "language": "en",
    }


def compute_file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================
# DATE EXTRACTION (CONTEXT AWARE)
# ============================================================

def extract_decision_date(text: str):
    """
    Extracts judgment/order date from header only.
    Looks for 'JUDGMENT DATED:' or 'DATED:' etc.
    """

    header = text[:1500]

    patterns = [
        r"(JUDGMENT\s+DATED\s*[:\-]\s*)(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
        r"(ORDER\s+DATED\s*[:\-]\s*)(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
        r"(DATED\s*[:\-]\s*)(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
        r"(DATE\s*[:\-]\s*)(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
    ]

    for pat in patterns:
        m = re.search(pat, header, re.IGNORECASE)
        if m:
            return m.group(2)

    return None


# ============================================================
# CASE NUMBER EXTRACTION (SAFE, DETERMINISTIC)
# ============================================================

def extract_case_number(text: str):
    """
    Court-safe extraction for High Court formats.
    Does NOT guess.
    """

    header = text[:1200]

    patterns = [
        r"R/FIRST APPEAL NO\.\s*\d+\s*OF\s*\d{4}",
        r"FIRST APPEAL NO\.\s*\d+\s*OF\s*\d{4}",
        r"C/FA/\d+/\d{4}",
    ]

    for pat in patterns:
        m = re.search(pat, header, re.IGNORECASE)
        if m:
            return m.group(0).strip()

    return None


# ============================================================
# LLM-BASED METADATA EXTRACTION (CONSERVATIVE)
# ============================================================

def extract_llm_metadata(text: str) -> dict:
    """
    Conservative judicial metadata extraction.
    LLM is allowed to classify, not invent.
    """

    prompt = f"""
You are a judicial document metadata extractor.

RULES:
- Extract ONLY if explicitly present
- Do NOT guess
- If 'ORDER' or 'ORAL ORDER' appears near the top → document_type = "Order"
- If 'JUDGMENT' or 'JUDGMENT DATED' appears near the top → document_type = "Judgment"

Return VALID JSON exactly in this format:
{{
  "domain": "Judicial | Government | Corporate | Academic | General",
  "document_type": "Order | Judgment | Petition | Reply | Notice | Other",
  "case_stage": "Interim | Final | Admission | Misc | Unknown",
  "petition_type": "MACP | Writ | Appeal | Criminal | Other | Unknown",
  "court_level": "High Court | District Court | Tribunal | Unknown",
  "case_number": null,
  "bench": null,
  "act_name": null
}}

Document Text:
----------------
{text[:4000]}
----------------
"""

    response = llm(
        prompt,
        temperature=0.0,
        max_tokens=300,
    )

    raw = response["choices"][0]["text"].strip()

    try:
        import json
        return json.loads(raw)
    except Exception:
        return {
            "domain": "General",
            "document_type": "Other",
            "case_stage": "Unknown",
            "petition_type": "Unknown",
            "court_level": "Unknown",
            "case_number": None,
            "bench": None,
            "act_name": None,
        }


# ============================================================
# SAVE METADATA TO DATABASE
# ============================================================

def save_document_metadata(
    document_id: str,
    filename: str | None,
    metadata: dict,
    user_id: str | None,
    collection_id: str | None,
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR REPLACE INTO document_metadata (
        document_id,
        filename,
        file_hash,
        page_count,
        word_count,
        language,
        document_type,
        case_stage,
        petition_type,
        act_name,
        court_level,
        case_number,
        order_date,
        bench,
        domain,
        user_id,
        collection_id,
        created_at,
        ingested_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        document_id,
        filename,
        metadata.get("file_hash"),
        metadata.get("page_count"),
        metadata.get("word_count"),
        metadata.get("language"),
        metadata.get("document_type"),
        metadata.get("case_stage"),
        metadata.get("petition_type"),
        metadata.get("act_name"),
        metadata.get("court_level"),
        metadata.get("case_number"),
        metadata.get("order_date"),
        metadata.get("bench"),
        metadata.get("domain"),
        user_id,
        collection_id,
        datetime.utcnow().isoformat(),
        datetime.utcnow().isoformat(),
    ))

    conn.commit()
    conn.close()


# ============================================================
# MAIN INGEST FUNCTION
# ============================================================

def ingest_document(
    text: str,
    document_id: str | None = None,
    collection_id: str | None = None,
    user_id: str | None = None,
    filename: str | None = None,
):
    """
    Ingests ONE document:
    - FAISS vectors
    - Judicial-grade metadata
    """

    if not document_id:
        document_id = str(uuid.uuid4())

    # ----------------------------
    # 1. SEMANTIC INGESTION
    # ----------------------------
    chunks = split_into_chunks(text)
    embeddings = embed_texts(chunks)

    vectors = [
        {"text": c, "embedding": e}
        for c, e in zip(chunks, embeddings)
    ]

    save_vectors(
        notebook_id=document_id,
        vectors=vectors,
        collection_id=collection_id,
    )

    # ----------------------------
    # 2. METADATA EXTRACTION
    # ----------------------------
    basic_meta = extract_basic_metadata(text)
    llm_meta = extract_llm_metadata(text)

    header_upper = text[:1000].upper()

    # Deterministic judicial overrides (SAFE)
    if "JUDGMENT" in header_upper:
        llm_meta["document_type"] = "Judgment"
        llm_meta["domain"] = "Judicial"

    elif "ORDER" in header_upper:
        llm_meta["document_type"] = "Order"
        llm_meta["domain"] = "Judicial"

    # Case number fallback (regex, not guessing)
    if not llm_meta.get("case_number"):
        llm_meta["case_number"] = extract_case_number(text)

    final_metadata = {
        **basic_meta,
        **llm_meta,
        "file_hash": compute_file_hash(text),
        "page_count": None,
        "order_date": extract_decision_date(text),
    }

    # ----------------------------
    # 3. SAVE METADATA
    # ----------------------------
    save_document_metadata(
        document_id=document_id,
        filename=filename,
        metadata=final_metadata,
        user_id=user_id,
        collection_id=collection_id,
    )

    return {
        "document_id": document_id,
        "chunks": len(chunks),
    }

