# rag/ingest.py
"""
Document ingestion pipeline.

This module extracts deterministic metadata, summarizes content, resolves legal
acts/sections, computes embeddings, and persists vectors + metadata to Postgres.
"""

import uuid
import json
import hashlib
import re
from typing import List, Dict
from datetime import datetime
from psycopg2.extras import Json

from rag.chunking import split_into_chunks
from rag.embedder import embed_texts
from rag.vector_store import save_vectors
from rag.model_router import qwen_summary
from rag.act_catalog import (
    extract_acts_with_sections,
    normalize_act_name as catalog_normalize_act_name,
    get_catalog_source,
)
from db import get_repo


# ============================================================
# PAGE ESTIMATION
# ============================================================

def estimate_page_count(text: str):
    if "\f" in text:
        return text.count("\f") + 1

    words = len(text.split())
    return max(1, round(words / 800))


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
# DETERMINISTIC EXTRACTORS
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

    if "SUPREME COURT OF INDIA" in header:
        return {
            "court_name": "Supreme Court of India",
            "court_level": "Supreme Court",
            "jurisdiction": {"country": "India"},
        }

    m = re.search(r"HIGH COURT OF\s+([A-Z ]+?)(?:\s+AT|\n|,|$)", header)
    if m:
        state = m.group(1).strip().title()
        return {
            "court_name": f"High Court of {state}",
            "court_level": "High Court",
            "jurisdiction": {"state": state},
        }

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


def extract_case_type(text):
    patterns = [
        "FIRST APPEAL",
        "CRIMINAL APPEAL",
        "SPECIAL CRIMINAL APPLICATION",
        "SPECIAL CRIMINAL APPLICATION (DIRECTION)",
        "SPECIAL CIVIL APPLICATION",
        "WRIT PETITION",
        "REVISION APPLICATION",
    ]
    header = text[:3000].upper()
    for p in patterns:
        if p in header:
            return p.title()
    return None


def extract_parties(text):
    m = re.search(
        r"\n([A-Z .()&]+)\n\s*Versus\s*\n([A-Z .()&]+)",
        text,
        re.IGNORECASE,
    )
    if m:
        return f"{m.group(1).strip()} vs {m.group(2).strip()}"
    return None


def extract_order_date(text):
    m = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", text[:2000])
    return normalize_date(m.group(0)) if m else None


def extract_judges_and_bench(text):
    header = text[:5000]
    m = re.search(r"(CORAM|BEFORE|PRESENT)\s*[:\-]?\s*(.*)", header, re.IGNORECASE)
    if not m:
        return [], []

    block = m.group(2)
    lines = re.split(r"[\n\r]+", block)

    judges = []
    buffer = ""

    for line in lines:
        clean = line.strip()
        if not clean:
            continue

        upper = clean.upper()

        if any(k in upper for k in ["DATE", "DATED", "ORAL", "JUDGMENT", "ORDER"]):
            break

        if "JUSTICE" in upper:
            if buffer:
                judges.append(buffer.strip())
            buffer = clean
        else:
            if buffer:
                buffer += " " + clean

    if buffer:
        judges.append(buffer.strip())

    final = []
    for j in judges:
        if "JUSTICE" not in j.upper():
            continue
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
    # Backward-compatible wrapper, now catalog-driven.
    return extract_acts_with_sections(text)



def extract_sections(text):
    sections = re.findall(r"Section\s+\d+[A-Za-z\-]*", text)
    return list(set(sections))


def normalize_and_dedup_acts(acts):
    seen = {}
    for a in acts:
        raw_name = a.get("act")
        canon = catalog_normalize_act_name(raw_name)
        if not canon:
            continue

        if canon not in seen:
            seen[canon] = {
                "act": canon,
                "sections": a.get("sections", [])
            }

    return list(seen.values())


def infer_acts_from_filename(filename: str | None) -> list[dict]:
    """
    Infer act candidates from filename/title when OCR body misses explicit Act marker.
    """
    if not filename:
        return []
    base = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    base = re.sub(r"[_\-]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    if not base:
        return []

    candidates = [
        base,
        re.sub(r"\bact\b", "Act", base, flags=re.IGNORECASE),
        re.sub(r"\bthe\b", "", base, flags=re.IGNORECASE).strip(),
    ]
    out = []
    seen = set()
    for c in candidates:
        canon = catalog_normalize_act_name(c)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        out.append({"act": canon, "sections": []})
    return out


def get_book_chunks(text: str) -> tuple[list[dict], list[str], str, list[str]]:
    """
    Build book chunks with TOC-first strategy, then legal-section fallback.
    Returns: (chunk_structs, chunk_texts, strategy, toc_entries)
    """
    from rag.chunker import chunk_by_sections, extract_toc_entries, chunk_by_toc

    toc_entries = extract_toc_entries(text)
    toc_chunks = chunk_by_toc(text, toc_entries)
    if toc_chunks:
        return toc_chunks, [c["full_text"] for c in toc_chunks], "toc", toc_entries

    section_chunks = chunk_by_sections(text, doc_type="book")
    return section_chunks, [c["full_text"] for c in section_chunks], "section", toc_entries


def extract_cited_cases(text):
    matches = re.findall(
        r"([A-Z][A-Za-z .&()]+ Vs\. [A-Z][A-Za-z .&()]+)",
        text
    )
    return list(set(matches))


def extract_final_directions(text):
    tail = text[-2500:].lower()
    if "appeal is allowed" in tail:
        return "Appeal Allowed"
    if "appeal is dismissed" in tail:
        return "Appeal Dismissed"
    if "deposit the said additional amount" in tail:
        return "Insurance Company directed to deposit compensation"
    return None


def extract_topics(text):
    topics = []
    lower = text.lower()

    if "motor accident" in lower:
        topics.append("Motor Accident Compensation")

    if "insurance company" in lower:
        topics.append("Insurance Liability")

    if "multiplier" in lower:
        topics.append("Multiplier Computation")

    if "compensation" in lower:
        topics.append("Compensation Award")

    return topics


def classify_document_kind(text: str, filename: str | None = None) -> str:
    """
    Remark: route metadata extraction by document kind.
    - `judgment` keeps existing case-law metadata structure.
    - `book` uses a dedicated schema for statutes/books/commentaries.
    """
    header = (text or "")[:6000].upper()
    name = (filename or "").upper()

    book_markers = [
        "TABLE OF CONTENTS",
        "CHAPTER",
        "PART I",
        "PART II",
        "BARE ACT",
        "SHORT TITLE, EXTENT AND COMMENCEMENT",
        "STATEMENT OF OBJECTS",
        "PRELIMINARY",
        "COMMENTARY",
        "CONSTITUTION OF INDIA",
    ]
    if any(marker in header for marker in book_markers):
        return "book"

    if any(k in name for k in ["ACT", "BARE", "CONSTITUTION", "COMMENTARY", "MANUAL"]):
        return "book"

    # Remark: require multiple strong signals for judicial classification.
    # Single words like "order" are too noisy for legal books.
    judicial_markers = [
        "JUDGMENT",
        "CORAM",
        "VERSUS",
        "IN THE HIGH COURT",
        "IN THE SUPREME COURT",
        "CRIMINAL APPEAL",
        "FIRST APPEAL",
        "WRIT PETITION",
        "SPECIAL CIVIL APPLICATION",
        "SPECIAL CRIMINAL APPLICATION",
    ]
    judicial_score = sum(1 for marker in judicial_markers if marker in header)
    if judicial_score >= 2:
        return "judgment"

    if re.search(r"\b[A-Z./]+\s*\d+/\d{4}\b", header):
        return "judgment"

    # Conservative fallback: unknown legal docs are treated as books to avoid
    # forcing case-specific metadata onto reference documents.
    return "book"


def extract_book_metadata(
    text: str,
    filename: str | None,
    basic: dict,
    page_count: int,
    toc_entries: list[str] | None = None,
):
    """
    Build book-specific metadata footprint for legal books/acts/reference docs.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    # Remark: skip OCR/page marker noise when selecting title.
    title = filename or "Untitled Document"
    for ln in lines[:80]:
        if re.match(r"^-+\s*PAGE\s+\d+\s*-+$", ln, flags=re.IGNORECASE):
            continue
        if re.match(r"^page\s+\d+\s+of\s+\d+$", ln, flags=re.IGNORECASE):
            continue
        if len(ln) < 3:
            continue
        title = ln[:250]
        break

    chapter_titles = []
    for ln in lines[:500]:
        if re.match(r"^(CHAPTER|PART)\s+[IVXLC0-9]+", ln, flags=re.IGNORECASE):
            chapter_titles.append(ln[:180])
    chapter_titles = list(dict.fromkeys(chapter_titles))[:100]

    sections = re.findall(r"\bSection\s+\d+[A-Za-z\-]*\b", text, flags=re.IGNORECASE)
    section_count = len(set(s.strip() for s in sections))
    articles = re.findall(r"\bArticle\s+\d+[A-Za-z\-]*\b", text, flags=re.IGNORECASE)
    article_count = len(set(a.strip() for a in articles))
    parts = re.findall(r"(?im)^\s*PART\s+[A-Z0-9IVXLC\-]+\b", text)
    part_count = len(set(p.strip().upper() for p in parts))
    schedules = re.findall(
        r"(?im)^\s*(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH|\d+(?:ST|ND|RD|TH)?)\s+SCHEDULE\b",
        text,
    )
    schedule_count = len(set(s.strip().upper() for s in schedules))

    acts = normalize_and_dedup_acts(extract_acts(text))
    inferred_from_name = infer_acts_from_filename(filename)
    if inferred_from_name:
        acts = normalize_and_dedup_acts(acts + inferred_from_name)
    act_alias_hits = [a["act"] for a in acts]
    toc_entries = toc_entries or []

    inferred_subjects = []
    low = text.lower()
    subject_markers = {
        "Constitutional Law": ["constitution", "fundamental rights", "directive principles"],
        "Criminal Law": ["criminal", "penal", "offence", "bnss", "bns"],
        "Civil Procedure": ["civil procedure", "cpc", "plaint", "decree"],
        "Evidence Law": ["evidence act", "admissible", "witness"],
        "Commercial Law": ["company", "insolvency", "contract", "commercial"],
    }
    for subject, markers in subject_markers.items():
        if any(m in low for m in markers):
            inferred_subjects.append(subject)

    confidence = 0.9 if title and (section_count > 0 or article_count > 0 or len(chapter_titles) > 0) else 0.7
    return {
        "title": title,
        "language": basic.get("language"),
        "source_type": basic.get("source_type"),
        "page_count": page_count,
        "word_count": basic.get("word_count"),
        "section_count": section_count,
        "article_count": article_count,
        "chapter_count": len(chapter_titles),
        "part_count": part_count,
        "schedule_count": schedule_count,
        "chapter_titles": chapter_titles,
        "act_alias_hits": act_alias_hits,
        "inferred_subjects": inferred_subjects,
        "structure_hints": {
            "has_tables": bool(basic.get("has_tables")),
            "has_annexures": bool(basic.get("has_annexures")),
            "has_citations": bool(basic.get("has_citations")),
            "has_toc": len(toc_entries) > 0,
        },
        "toc_count": len(toc_entries),
        "toc_entries": toc_entries[:300],
        "metadata_confidence": round(confidence, 2),
        "extraction_notes": {
            "source": "book-profile",
            "acts_catalog_source": get_catalog_source(),
            "structure_counts": {
                "sections": section_count,
                "articles": article_count,
                "parts": part_count,
                "schedules": schedule_count,
                "toc_entries": len(toc_entries),
            },
        },
    }


def _extract_sub_units_from_chunk(section_code: str, section_text: str) -> List[Dict]:
    """
    Extract subsection/clause markers from a section chunk.
    """
    out: List[Dict] = []
    base = (section_code or "").strip().upper()
    if not base or not section_text:
        return out

    seen = set()

    current_subsection = None

    # Subsections: (1), (2), ...
    for m in re.finditer(r"\((\d{1,3})\)", section_text):
        n = m.group(1)
        code = f"{base}({n})"
        if code in seen:
            continue
        seen.add(code)
        out.append({"section_code": code, "section_type": "subsection", "parent_section_code": base})
        current_subsection = code

    # Clauses: (a), (b), ...
    for m in re.finditer(r"\(([a-zA-Z]{1,3})\)", section_text):
        c = m.group(1).upper()
        if c.isdigit():
            continue
        # Attach clause to latest subsection when present, else to base section.
        parent = current_subsection or base
        code = f"{parent}({c})"
        if code in seen:
            continue
        seen.add(code)
        out.append({"section_code": code, "section_type": "clause", "parent_section_code": parent})

    return out


def _tokenize_index_match(text: str | None) -> List[str]:
    s = (text or "").lower()
    s = s.replace(".pdf", " ")
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t]
    stop = {"the", "of", "and", "act", "code", "sanhita", "adhiniyam", "law", "rules"}
    out = []
    for t in toks:
        if t in stop:
            continue
        if len(t) > 4 and t.endswith("s"):
            t = t[:-1]
        out.append(t)
    return out


def _pick_primary_acts_for_section_index(
    filename: str | None,
    act_alias_hits: List[str],
) -> List[str]:
    """
    Choose stable canonical act(s) for section index rows.

    Why:
    - Full-text act extraction for books captures many cross-references.
    - Indexing each section under all mentioned acts corrupts lookup quality.
    """
    acts = [a for a in (act_alias_hits or []) if a]
    if not acts:
        return []

    # Strongest signal: filename-derived act alias.
    fname_acts = []
    for a in infer_acts_from_filename(filename):
        if isinstance(a, dict):
            name = a.get("act")
        else:
            name = a
        if name:
            fname_acts.append(name)
    if fname_acts:
        return list(dict.fromkeys(fname_acts))

    # Fallback: pick best filename-token overlap from detected acts.
    file_tokens = set(_tokenize_index_match(filename))
    if file_tokens:
        scored = []
        for act in acts:
            act_tokens = set(_tokenize_index_match(act))
            if not act_tokens:
                continue
            overlap = len(file_tokens.intersection(act_tokens))
            scored.append((overlap, act))
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored and scored[0][0] > 0:
            best = [a for s, a in scored if s == scored[0][0]]
            # Keep deterministic + bounded.
            return list(dict.fromkeys(best[:2]))

    # Last resort: first canonical hit only.
    return [acts[0]]


def build_book_section_rows(
    document_id: str,
    filename: str | None,
    user_id: str | None,
    collection_id: str | None,
    section_chunks: List[Dict],
    act_alias_hits: List[str],
) -> List[Dict]:
    """
    Build normalized Act+Section rows for fast disambiguated retrieval.
    """
    rows: List[Dict] = []
    # Remark: do NOT fan-out sections to every referenced act in the text.
    # This table must represent owning act(s) of the book, not citations.
    acts = _pick_primary_acts_for_section_index(filename, act_alias_hits)
    if not acts:
        return rows

    for idx, chunk in enumerate(section_chunks):
        section_id = (chunk.get("section_id") or "").strip()
        if not section_id:
            continue
        section_type = (chunk.get("section_type") or "section").strip().lower()
        section_code = section_id.upper()
        section_title = (chunk.get("section_title") or f"Section {section_id}")[:300]
        full_chunk_text = (chunk.get("full_text") or chunk.get("text") or "")
        preview = full_chunk_text[:600]

        for act in acts:
            rows.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "user_id": user_id,
                    "collection_id": collection_id,
                    "act_canonical": act,
                    "section_code": section_code,
                    "parent_section_code": None,
                    "section_type": section_type,
                    "section_title": section_title,
                    "chunk_index": idx,
                    "text_preview": preview,
                }
            )
            # Also index subsection/clause paths for more precise retrieval.
            for sub in _extract_sub_units_from_chunk(section_code, full_chunk_text):
                rows.append(
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "user_id": user_id,
                        "collection_id": collection_id,
                        "act_canonical": act,
                        "section_code": sub["section_code"],
                        "parent_section_code": sub["parent_section_code"],
                        "section_type": sub["section_type"],
                        "section_title": f"Section {sub['section_code']}",
                        "chunk_index": idx,
                        "text_preview": preview,
                    }
                )
    return rows


def document_length_class(word_count):
    if word_count < 2000:
        return "Short"
    if word_count < 8000:
        return "Medium"
    return "Long"


# ============================================================
# LLM SHORT ABSTRACT (CONTROLLED)
# ============================================================

def generate_short_abstract(text, meta):

    role = meta.get("document_role")

    if role == "Judicial":

        structured_context = f"""
Case Type: {meta.get("case_type")}
Court: {meta.get("court_name")}
Parties: {meta.get("parties")}
Acts: {', '.join([a['act'] for a in meta.get('act_names', [])])}
Decision: {meta.get("decision_status")}
"""

        prompt = f"""
You are a judicial abstract generator.

RULES:
- Max 100 words.
- No speculation.
- Focus on legal issue and outcome.
- Professional tone.

STRUCTURED METADATA:
{structured_context}

DOCUMENT EXCERPT:
{text[:3000]}

Generate concise legal abstract.
"""

    else:

        prompt = f"""
You are a professional document summarizer.

RULES:
- Max 80 words.
- Neutral tone.
- No speculation.

DOCUMENT:
{text[:3000]}

Generate short description.
"""

    try:
        return qwen_summary(prompt, max_tokens=180, temperature=0.15)
    except:
        return None


def _derive_extraction_status(
    role: str | None,
    metadata_confidence: float,
    act_count: int,
):
    # For judicial documents, act extraction is important for downstream analytics.
    if role == "Judicial" and act_count == 0:
        if metadata_confidence < 0.6:
            return "failed", True
        return "needs_review", True

    if metadata_confidence >= 0.85:
        return "complete", False
    if metadata_confidence >= 0.6:
        return "needs_review", True
    return "failed", True


# ============================================================
# MAIN INGEST
# ============================================================

def ingest_document(
    text: str,
    document_id: str | None = None,
    collection_id: str | None = None,
    user_id: str | None = None,
    filename: str | None = None,
    pdf_page_count: int | None = None,
):

    if not document_id:
        document_id = str(uuid.uuid4())

    # ---- Deterministic extraction FIRST to determine chunking strategy ----
    basic = extract_basic(text)
    if isinstance(pdf_page_count, int) and pdf_page_count > 0:
        page_count = pdf_page_count
    else:
        page_count = estimate_page_count(text)
    
    # Classify document kind to decide chunking strategy
    doc_kind = classify_document_kind(text, filename)
    role = "Judicial" if doc_kind == "judgment" else "ReferenceBook"
    
    # ---- INTELLIGENT CHUNKING STRATEGY ----
    # For legal reference books/acts → use section-aware chunking
    # For judgments/cases → use paragraph-based chunking
    section_chunks = []
    chunk_strategy = "paragraph"
    toc_entries: list[str] = []
    if role == "ReferenceBook":
        section_chunks, chunks, chunk_strategy, toc_entries = get_book_chunks(text)
        print(f"📖 Using {chunk_strategy}-aware chunking: {len(chunks)} chunks detected")
    else:
        chunks = split_into_chunks(text)
        print(f"📄 Using paragraph-based chunking: {len(chunks)} chunks")
    
    # ---- Vector ingestion ----
    embeddings = embed_texts(chunks)
    vectors_payload = []
    for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
        payload = {"text": chunk_text, "embedding": emb, "chunk_index": i}
        if role == "ReferenceBook" and i < len(section_chunks):
            payload["section_id"] = section_chunks[i].get("section_id")
            payload["section_title"] = section_chunks[i].get("section_title")
            payload["section_type"] = section_chunks[i].get("section_type")
        vectors_payload.append(payload)

    save_vectors(
        notebook_id=document_id,
        vectors=vectors_payload,
        collection_id=collection_id,
    )

    # ---- REST OF METADATA EXTRACTION ----
    court = extract_court(text) if doc_kind == "judgment" else {}

    case_no = extract_case_number(text) if doc_kind == "judgment" else None
    case_type = extract_case_type(text) if doc_kind == "judgment" else None
    parties = extract_parties(text) if doc_kind == "judgment" else None
    order_dt = extract_order_date(text) if doc_kind == "judgment" else None
    judges, bench = extract_judges_and_bench(text) if doc_kind == "judgment" else ([], [])
    doc_type, case_stage = infer_document_type_and_stage(text) if doc_kind == "judgment" else ("Book", None)
    decision_status = extract_decision_status(text) if doc_kind == "judgment" else None

    raw_acts = extract_acts(text)
    if role == "ReferenceBook":
        raw_acts = (raw_acts or []) + infer_acts_from_filename(filename)
    acts = normalize_and_dedup_acts(raw_acts)

    cited_cases = extract_cited_cases(text)
    final_directions = extract_final_directions(text)
    primary_topics = extract_topics(text)

    metadata = {
        "document_id": document_id,
        "filename": filename,
        "user_id": user_id,
        "collection_id": collection_id,
        "document_role": role,
        "domain": "Judicial" if role == "Judicial" else "Reference",
        "domain_confidence": 0.9 if role == "Judicial" else 0.85,
        **basic,
        "page_count": page_count,
        **court,
        "case_number": case_no,
        "case_type": case_type,
        "parties": parties,
        "order_date": order_dt,
        "judge_name": judges,
        "bench": bench,
        "document_type": doc_type,
        "case_stage": case_stage,
        "decision_status": decision_status,
        "outcome": decision_status,
        "act_names": acts,
        "referenced_laws": [a["act"] for a in acts],
        "cited_cases": cited_cases,
        "final_directions": final_directions,
        "primary_topics": primary_topics if doc_kind == "judgment" else [],
        "document_length_class": document_length_class(basic["word_count"]),
        "file_hash": sha256(text),
        "created_at": datetime.utcnow().isoformat(),
        "ingested_at": datetime.utcnow().isoformat(),
        "intended_audience": "Mixed",
    }

    # Basic confidence footprints so these fields are not left null.
    confidence_fields = {
        "court_name": metadata.get("court_name"),
        "case_number": metadata.get("case_number"),
        "judge_name": metadata.get("judge_name"),
        "order_date": metadata.get("order_date"),
        "decision_status": metadata.get("decision_status"),
        "document_type": metadata.get("document_type"),
        "act_names": metadata.get("act_names"),
        "primary_topics": metadata.get("primary_topics"),
    }
    filled = 0
    field_confidence = {}
    for key, value in confidence_fields.items():
        ok = bool(value) if not isinstance(value, list) else len(value) > 0
        field_confidence[key] = 0.9 if ok else 0.0
        if ok:
            filled += 1

    metadata["field_confidence"] = field_confidence
    metadata_conf = round(filled / len(confidence_fields), 2)
    metadata["metadata_confidence"] = metadata_conf
    extraction_status, needs_review = _derive_extraction_status(
        role=metadata.get("document_role"),
        metadata_confidence=metadata_conf,
        act_count=len(metadata.get("act_names") or []),
    )
    if doc_kind == "book":
        # Remark: book metadata follows dedicated table/profile and should not
        # enter judicial retry loop.
        extraction_status, needs_review = "complete", False
    metadata["extraction_status"] = extraction_status
    metadata["needs_review"] = needs_review
    metadata["retry_count"] = 0
    metadata["last_retry_at"] = None
    metadata["extraction_notes"] = {
        "page_count": "pdf_page_count if available; fallback to text estimate",
        "word_count": "deterministic split() count",
        "acts_catalog_source": get_catalog_source(),
        "missing_fields": [k for k, v in field_confidence.items() if v == 0.0],
    }

    metadata["document_about"] = generate_short_abstract(text, metadata)

    repo = get_repo()
    repo.insert_document(metadata)

    if doc_kind == "book":
        # Remark: persist richer book-focused metadata for legal books/acts.
        book_meta = extract_book_metadata(
            text=text,
            filename=filename,
            basic=basic,
            page_count=page_count,
            toc_entries=toc_entries,
        )
        book_meta.update(
            {
                "document_id": document_id,
                "filename": filename,
                "user_id": user_id,
                "collection_id": collection_id,
                "file_hash": metadata.get("file_hash"),
                "created_at": metadata.get("created_at"),
                "ingested_at": metadata.get("ingested_at"),
            }
        )
        repo.insert_book_document(book_meta)
        section_rows = build_book_section_rows(
            document_id=document_id,
            filename=filename,
            user_id=user_id,
            collection_id=collection_id,
            section_chunks=section_chunks,
            act_alias_hits=book_meta.get("act_alias_hits") or [],
        )
        repo.upsert_book_section_rows(document_id, section_rows)

    # Queue low-confidence metadata for async retry refinement.
    if needs_review:
        try:
            cur = repo.conn.cursor()
            max_queued = 50
            if user_id:
                cur.execute(
                    """
                    SELECT value
                    FROM admin_runtime_settings
                    WHERE key = 'metadata_retry_max_queued_per_user'
                    """
                )
                cap_row = cur.fetchone()
                if cap_row:
                    try:
                        max_queued = max(1, int(cap_row[0]))
                    except Exception:
                        max_queued = 50
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM metadata_retry_jobs
                    WHERE user_id = %s
                      AND status IN ('queued', 'processing')
                    """,
                    (user_id,),
                )
                queued_for_user = int(cur.fetchone()[0] or 0)
                if queued_for_user >= max_queued:
                    notes = metadata.get("extraction_notes") or {}
                    notes["retry_queue_skipped"] = f"user queue cap reached ({max_queued})"
                    cur.execute(
                        """
                        UPDATE document_metadata
                        SET extraction_notes = %s
                        WHERE document_id = %s
                        """,
                        (Json(notes), document_id),
                    )
                    repo.conn.commit()
                    return {
                        "document_id": document_id,
                        "chunks": len(chunks),
                    }

            cur.execute(
                """
                INSERT INTO metadata_retry_jobs (
                    document_id, user_id, status, attempts, next_retry_at, last_error, created_at, updated_at
                )
                VALUES (%s, %s, 'queued', 0, NOW(), NULL, NOW(), NOW())
                ON CONFLICT (document_id) DO UPDATE
                SET status='queued', next_retry_at=NOW(), updated_at=NOW()
                """,
                (document_id, user_id),
            )
            repo.conn.commit()
        except Exception:
            # Do not fail ingest due to retry queue write issues.
            pass

    return {
        "document_id": document_id,
        "chunks": len(chunks),
    }

def normalize_act_name(name: str):
    return catalog_normalize_act_name(name)
