# rag/rag_pipeline.py
"""
Hybrid RAG orchestration for notebook, collection, and global scopes.

Flow:
1) Fast metadata route for count/list/page questions
2) Section-first retrieval for legal acts (CPC, BNS, IPC, etc.)
3) Intent routing for answer strategy
4) Retrieval from notebook/collection/global indexes
5) Final LLM synthesis
"""

import numpy as np
import faiss
import re
from typing import List, Dict, Optional

from rag.llm import ask_llm, llm_generate_text
from rag.model_router import qwen_summary
from rag.vector_store import (
    load_vectors,
    load_global_index,
    get_collection_notebooks,
)
from rag.embedder import embed_texts
from rag.section_retrieval import (
    retrieve_by_section_first,
    find_act_for_document,
    find_documents_with_act,
    extract_acts_from_question,
)
from rag.chunker import extract_all_sections

from rag.question_router import classify_question


def _chunk_preview(text: str, limit: int = 180) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _score_from_distance(distance: float) -> float:
    return round(1.0 / (1.0 + max(distance, 0.0)), 4)


def _make_response(answer: str, citations=None, mode: str = "rag"):
    return {
        "answer": answer,
        "citations": citations or [],
        "mode": mode,
    }


def _append_scored_chunk(scored_chunks, distance: float, chunk_meta: dict, source_type: str):
    text = (chunk_meta or {}).get("text") or ""
    if not text.strip():
        return

    scored_chunks.append(
        {
            "distance": float(distance),
            "score": _score_from_distance(float(distance)),
            "text": text,
            "notebook_id": (chunk_meta or {}).get("notebook_id"),
            "collection_id": (chunk_meta or {}).get("collection_id"),
            "chunk_index": (chunk_meta or {}).get("chunk_index"),
            "source_type": source_type,
        }
    )


def _select_top_unique_chunks(scored_chunks, max_total_chunks: int):
    scored_chunks.sort(key=lambda x: x["distance"])
    deduped = []
    seen = set()
    for chunk in scored_chunks:
        text = chunk["text"]
        if text in seen:
            continue
        seen.add(text)
        deduped.append(chunk)
        if len(deduped) >= max_total_chunks:
            break
    return deduped


def _build_citations(chunks, max_citations: int = 6):
    citations = []
    for chunk in chunks[:max_citations]:
        citations.append(
            {
                "source_type": chunk.get("source_type"),
                "notebook_id": chunk.get("notebook_id"),
                "collection_id": chunk.get("collection_id"),
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score"),
                "preview": _chunk_preview(chunk.get("text", "")),
            }
        )
    return citations


def _normalize_collection_filters(global_sub_collection_ids) -> List[str]:
    ids = []
    seen = set()
    for raw in (global_sub_collection_ids or []):
        cid = str(raw or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        ids.append(cid)
    return ids


def _allow_global_chunk(chunk_meta: dict, allowed_collections: set[str]) -> bool:
    if not allowed_collections:
        return True
    cid = str((chunk_meta or {}).get("collection_id") or "").strip()
    return bool(cid and cid in allowed_collections)


def _lookup_collection_ids_for_notebooks(notebook_ids: List[str]) -> Dict[str, str]:
    if not notebook_ids:
        return {}
    try:
        from db import get_repo
        repo = get_repo()
        cur = repo.conn.cursor()
        placeholders = ",".join(["%s"] * len(notebook_ids))
        cur.execute(
            f"""
            SELECT notebook_id, collection_id
            FROM notebooks
            WHERE notebook_id IN ({placeholders})
            """,
            tuple(notebook_ids),
        )
        return {row[0]: row[1] for row in (cur.fetchall() or [])}
    except Exception:
        return {}


def _filter_section_hits_by_global_subcollections(section_chunks: list[dict], allowed_ids: List[str]) -> list[dict]:
    if not section_chunks or not allowed_ids:
        return section_chunks
    allowed = set(allowed_ids)
    notebook_ids = [str(c.get("notebook_id") or "") for c in section_chunks if c.get("notebook_id")]
    nb_map = _lookup_collection_ids_for_notebooks(notebook_ids)
    return [c for c in section_chunks if nb_map.get(str(c.get("notebook_id") or "")) in allowed]


def _specialization_instructions(specialization: str | None) -> str:
    spec = (specialization or "").strip().lower()
    if not spec or spec in {"general", "default"}:
        return ""
    profile = _get_specialization_profile(spec)
    if profile and profile.get("instruction"):
        return f"[Specialization Mode]\n{str(profile.get('instruction')).strip()}\n"
    templates = {
        "criminal": "Focus on criminal law procedure, ingredients of offence, burden of proof, and judicial tests.",
        "criminal_law": "Focus on criminal law procedure, ingredients of offence, burden of proof, and judicial tests.",
        "civil": "Focus on civil remedies, pleadings, limitation, maintainability, and decree/enforcement implications.",
        "civil_law": "Focus on civil remedies, pleadings, limitation, maintainability, and decree/enforcement implications.",
        "constitutional": "Focus on constitutional principles, fundamental rights, proportionality, and precedent hierarchy.",
        "tax": "Focus on charging provisions, exemptions, classification, and ratio of cited tax precedents.",
        "taxation": "Focus on charging provisions, exemptions, classification, and ratio of cited tax precedents.",
        "evidence": "Focus on admissibility, relevancy, presumptions, burden shifts, and evidentiary value.",
        "procedural": "Focus on procedural compliance, timelines, jurisdiction, and defects curable/incurable.",
    }
    instruction = templates.get(spec, f"Answer with strong {specialization} specialization and legal precision.")
    return f"[Specialization Mode]\n{instruction}\n"


def _normalize_act_mentions(act_mentions: List[str]) -> List[str]:
    out = []
    seen = set()
    for raw in (act_mentions or []):
        act = str(raw or "").strip()
        if not act or act in seen:
            continue
        seen.add(act)
        out.append(act)
    return out


def _document_has_any_act(document_id: str | None, act_mentions: List[str]) -> bool:
    if not document_id or not act_mentions:
        return False
    aliases = {str(a or "").strip().lower() for a in (find_act_for_document(document_id) or [])}
    if not aliases:
        return False
    for act in act_mentions:
        if act.lower() in aliases:
            return True
    return False


def _needs_procedural_guidance(question: str) -> bool:
    q = (question or "").lower()
    if not q:
        return False
    markers = [
        "next probable step",
        "next probable steps",
        "next step",
        "next stage",
        "probable outcome",
        "likely outcome",
        "what is lacking",
        "what is missing",
        "what should be done next",
    ]
    return any(m in q for m in markers)


def _procedural_guidance_instruction(question: str) -> str:
    if not _needs_procedural_guidance(question):
        return ""
    return (
        "[Procedural Guidance Mode]\n"
        "Provide: (1) likely next legal steps, (2) probable outcomes with conditions, "
        "(3) what facts/documents are missing, and (4) immediate practical actions. "
        "Use cautious legal wording, not certainty.\n"
    )


def _get_specialization_profile(specialization: str | None) -> dict:
    spec = (specialization or "").strip().lower()
    if not spec:
        return {}
    try:
        from db import get_repo
        repo = get_repo()
        cur = repo.conn.cursor()
        cur.execute(
            """
            SELECT config
            FROM specialization_profiles
            WHERE key = %s AND is_active = TRUE
            """,
            (spec,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        cfg = row[0] or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _is_ni_act_context(
    question: str,
    act_mentions: List[str],
    specialization: Optional[str],
) -> bool:
    spec = (specialization or "").strip().lower()
    if spec in {"ni_act", "ni", "negotiable_instruments"}:
        return True
    if any("negotiable" in (a or "").lower() and "instrument" in (a or "").lower() for a in (act_mentions or [])):
        return True
    q = (question or "").lower()
    return bool(re.search(r"\bni\s*act\b|\bnegotiable\s+instruments?\s+act\b", q))


def _ni_act_guidance_instruction(
    question: str,
    act_mentions: List[str],
    specialization: Optional[str],
) -> str:
    if not _is_ni_act_context(question, act_mentions, specialization):
        return ""

    profile = _get_specialization_profile("ni_act")
    stages = profile.get("stages") if isinstance(profile.get("stages"), list) else []
    presumptions = profile.get("presumptions") if isinstance(profile.get("presumptions"), list) else []
    missing = profile.get("missing_info_checklist") if isinstance(profile.get("missing_info_checklist"), list) else []

    stage_lines = "\n".join([f"- {str(s).strip()}" for s in stages[:12]]) if stages else "- Complaint stage mapping unavailable"
    presumption_lines = "\n".join([f"- {str(p).strip()}" for p in presumptions[:8]]) if presumptions else "- Presumption profile unavailable"
    missing_lines = "\n".join([f"- {str(m).strip()}" for m in missing[:12]]) if missing else "- Missing-info checklist unavailable"

    return (
        "[NI Act Engine]\n"
        "When applicable, answer in this structure:\n"
        "1) Current stage\n"
        "2) Probable next step(s)\n"
        "3) Probable outcome(s) with conditions\n"
        "4) Missing facts/documents\n"
        "5) Risk note\n"
        "Use NI Act presumptions carefully and mention they are rebuttable.\n"
        "Reference stage model:\n"
        f"{stage_lines}\n"
        "Reference presumptions:\n"
        f"{presumption_lines}\n"
        "Reference missing-info checklist:\n"
        f"{missing_lines}\n"
    )


def _build_section_exact_answer(
    question: str,
    section_chunks: list[dict],
    llm_question: str | None = None,
) -> str:
    """
    Deterministic formatter for exact section hits.
    Avoids LLM drift like "Not mentioned" when exact section text is already found.
    """
    if not section_chunks:
        return "Not mentioned in the documents."

    q = (question or "").lower()
    wants_explain = any(k in q for k in ["explain", "interpret", "meaning", "in simple", "summary", "summarize"])

    lines = []
    seen = set()
    for chunk in section_chunks[:3]:
        section_id = str(chunk.get("section_id") or "").strip()
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        text = _extract_exact_section_text(text, section_id)
        key = (section_id, text[:180])
        if key in seen:
            continue
        seen.add(key)
        heading = f"Section {section_id}" if section_id else "Section"
        snippet = " ".join(text.split())
        if len(snippet) > 900:
            snippet = snippet[:900].rstrip() + "..."
        lines.append(f"{heading}: {snippet}")

    if not lines:
        return "Not mentioned in the documents."

    if wants_explain:
        # Keep explanation mode enabled for descriptive asks.
        context = "\n\n".join((c.get("text") or "") for c in section_chunks[:3])
        explained = ask_llm(context[:7000], llm_question or question)
        if explained and "not mentioned in the document" not in explained.lower():
            return explained

    if len(lines) == 1:
        return lines[0]
    return "Relevant exact provisions:\n" + "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines))


def _extract_exact_section_text(text: str, section_id: str) -> str:
    """
    Trim chunk text to the requested section span when nearby OCR includes
    adjacent section headings or boilerplate.
    """
    raw = " ".join((text or "").split())
    if not raw:
        return ""
    sec = (section_id or "").strip()
    if not sec:
        return raw[:1200]

    # Start at the matched section heading if present.
    start_pat = re.compile(
        rf"\b(?:section\s+)?{re.escape(sec)}\s*[\.\):\-]",
        flags=re.IGNORECASE,
    )
    m = start_pat.search(raw)
    start = m.start() if m else 0
    tail = raw[start:]

    # End before next numbered section heading.
    next_pat = re.compile(
        r"\s(?:section\s+)?\d{1,4}[A-Za-z]?\s*[\.\):\-]\s",
        flags=re.IGNORECASE,
    )
    m2 = next_pat.search(tail[1:])  # skip immediate heading at start
    if m2:
        end = 1 + m2.start()
        tail = tail[:end]

    # Drop common Gazette/page artifacts.
    tail = re.sub(r"\bTHE\s+GAZETTE\s+OF\s+INDIA\b.*$", "", tail, flags=re.IGNORECASE)
    tail = re.sub(r"\bSec\.\s*\d+\b.*$", "", tail, flags=re.IGNORECASE)
    tail = re.sub(r"\s{2,}", " ", tail).strip(" -:\n\t")
    if len(tail) > 1300:
        tail = tail[:1300].rstrip() + "..."
    return tail


def _enrich_citation_filenames(citations, user_id=None):
    if not citations:
        return citations

    notebook_ids = sorted(
        {
            c.get("notebook_id")
            for c in citations
            if c.get("notebook_id")
        }
    )
    if not notebook_ids:
        return citations

    try:
        from db import get_repo
        repo = get_repo()
        cur = repo.conn.cursor()

        placeholders = ",".join(["%s"] * len(notebook_ids))
        if user_id:
            cur.execute(
                f"""
                SELECT notebook_id, filename
                FROM notebooks
                WHERE notebook_id IN ({placeholders})
                  AND (user_id = %s OR user_id IS NULL)
                """,
                tuple(notebook_ids + [user_id]),
            )
        else:
            cur.execute(
                f"""
                SELECT notebook_id, filename
                FROM notebooks
                WHERE notebook_id IN ({placeholders})
                """,
                tuple(notebook_ids),
            )

        filename_map = {row[0]: row[1] for row in cur.fetchall()}
        for citation in citations:
            nb_id = citation.get("notebook_id")
            if nb_id and filename_map.get(nb_id):
                citation["filename"] = filename_map[nb_id]
    except Exception:
        return citations

    return citations


def _lookup_notebook_filenames(notebook_ids, user_id=None):
    if not notebook_ids:
        return {}
    try:
        from db import get_repo
        repo = get_repo()
        cur = repo.conn.cursor()
        placeholders = ",".join(["%s"] * len(notebook_ids))
        if user_id:
            cur.execute(
                f"""
                SELECT notebook_id, filename
                FROM notebooks
                WHERE notebook_id IN ({placeholders})
                  AND (user_id = %s OR user_id IS NULL)
                """,
                tuple(list(notebook_ids) + [user_id]),
            )
        else:
            cur.execute(
                f"""
                SELECT notebook_id, filename
                FROM notebooks
                WHERE notebook_id IN ({placeholders})
                """,
                tuple(notebook_ids),
            )
        return {row[0]: row[1] for row in cur.fetchall()}
    except Exception:
        return {}


def _is_generic_ambiguous_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    # Short/open-ended prompts are typically under-specified in multi-notebook scopes.
    if len(q.split()) <= 8:
        return True
    vague_markers = [
        "what is this",
        "explain this",
        "tell me about this",
        "what about this",
        "is it",
        "can you explain",
        "give summary",
        "summarize",
        "details",
    ]
    return any(m in q for m in vague_markers)


def _maybe_return_collection_clarification(question, selected_chunks, user_id=None):
    """
    Ask for clarification when retrieval spans multiple notebooks and the
    question is under-specified.
    """
    if not selected_chunks or not _is_generic_ambiguous_question(question):
        return None

    notebook_ids = [c.get("notebook_id") for c in selected_chunks if c.get("notebook_id")]
    unique_notebooks = list(dict.fromkeys(notebook_ids))
    if len(unique_notebooks) <= 1:
        return None

    # If top-ranked chunks are all from one notebook, avoid unnecessary prompts.
    top_sources = {c.get("notebook_id") for c in selected_chunks[:4] if c.get("notebook_id")}
    if len(top_sources) == 1:
        return None

    name_map = _lookup_notebook_filenames(unique_notebooks[:8], user_id=user_id)
    options = []
    for nb in unique_notebooks[:6]:
        nm = name_map.get(nb)
        if nm:
            options.append(nm)
    hint = f" Candidate notebooks: {'; '.join(options)}." if options else ""
    return _make_response(
        "Your question can map to multiple notebooks in this collection. Please specify which notebook or ask to compare across all." + hint,
        citations=[],
        mode="collection_ambiguous",
    )


def _detect_summary_request(question: str):
    q = (question or "").strip().lower()
    if not q:
        return {
            "is_summary": False,
            "min_words": None,
            "max_words": None,
            "target_words": None,
        }

    summary_markers = [
        "summary",
        "summarize",
        "summarise",
        "describe",
        "described",
        "description",
        "explain",
        "explainer",
        "briefly explain",
        "in brief",
        "short note",
        "notes",
        "key points",
        "highlights",
        "synopsis",
        "abstract",
        "overview",
        "gist",
        "tl dr",
        "tldr",
        "description",
        "brief",
    ]
    is_summary = any(marker in q for marker in summary_markers)

    min_words = None
    max_words = None
    target_words = None

    # e.g. "100 to 200 words", "100-200 words", "100 or 200 words"
    range_match = re.search(
        r"\b(\d{2,4})\s*(?:-|to|or)\s*(\d{2,4})\s*words?\b",
        q,
        flags=re.IGNORECASE,
    )
    if range_match:
        a = int(range_match.group(1))
        b = int(range_match.group(2))
        lo, hi = sorted([a, b])
        min_words = lo
        max_words = hi
        target_words = round((lo + hi) / 2)
    else:
        # e.g. "in 150 words"
        exact_match = re.search(r"\b(?:in|within|around|about)?\s*(\d{2,4})\s*words?\b", q, flags=re.IGNORECASE)
        if exact_match:
            n = int(exact_match.group(1))
            min_words = max(10, n - 15)
            max_words = n + 15
            target_words = n

    # Word-count bounded asks should use summary pipeline even if marker is missing.
    if min_words is not None or max_words is not None or target_words is not None:
        is_summary = True

    # Clamp to safe window
    if min_words is not None:
        min_words = max(20, min(min_words, 1200))
    if max_words is not None:
        max_words = max(40, min(max_words, 1400))
    if target_words is not None:
        target_words = max(20, min(target_words, 1200))

    return {
        "is_summary": is_summary,
        "min_words": min_words,
        "max_words": max_words,
        "target_words": target_words,
    }


def _maybe_expand_section_followup_from_history(
    question: str,
    *,
    collection_id: str | None,
    notebook_id: str | None,
    user_id: str | None,
) -> str:
    """
    Safety net for follow-ups like "NI Act" after "What is Section 38?" ambiguity.
    """
    q = (question or "").strip()
    if not q or extract_all_sections(q):
        return question
    if len(q) > 180 or len(q.split()) > 12:
        return question
    if not user_id:
        return question
    if not collection_id and not notebook_id:
        return question
    try:
        from db import get_repo
        repo = get_repo()
        cur = repo.conn.cursor()
        if collection_id:
            cur.execute(
                """
                SELECT role, content
                FROM collection_chat_history
                WHERE collection_id = %s AND user_id = %s
                ORDER BY id DESC
                LIMIT 14
                """,
                (collection_id, user_id),
            )
        else:
            cur.execute(
                """
                SELECT role, content
                FROM chat_history
                WHERE notebook_id = %s
                ORDER BY id DESC
                LIMIT 14
                """,
                (notebook_id,),
            )
        rows = cur.fetchall() or []
        if not rows:
            return question

        latest_assistant = next((r[1] for r in rows if (r[0] or "").lower() == "assistant"), "") or ""
        low_assistant = latest_assistant.lower()
        is_disambiguation_reply = (
            "please specify the book name" in low_assistant
            or "multiple acts contain that section number" in low_assistant
        )

        latest_user_with_section = None
        latest_user_with_section_text = ""
        users_desc = [(r, c) for r, c in rows if (r or "").lower() == "user"]
        for _, content in users_desc:
            secs = extract_all_sections(content or "")
            if secs:
                latest_user_with_section = secs
                latest_user_with_section_text = content or ""
                break
        if not latest_user_with_section:
            return question

        # Case A: ambiguity follow-up ("BNSS", "NI Act").
        if is_disambiguation_reply:
            if len(latest_user_with_section) == 1:
                num, typ = latest_user_with_section[0]
                return f"What is {typ.title()} {num} of {q}?"
            nums = [num for num, _ in latest_user_with_section[:4]]
            joined = ", ".join(nums[:-1]) + (f" and {nums[-1]}" if len(nums) > 1 else nums[0])
            return f"What are Sections {joined} of {q}?"

        # Case B: referential follow-up ("give gist of same/this/it").
        low_q = q.lower()
        referential = bool(re.search(r"\b(same|this|that|it|above|previous|earlier)\b", low_q))
        summary_like = any(k in low_q for k in ["gist", "summary", "summarize", "summarise", "describe", "explain"])
        if referential and summary_like:
            sec_num, sec_typ = latest_user_with_section[0]
            # Pick latest short user phrase that likely denotes act/book (e.g. BNSS, NI Act).
            act_hint = None
            for _, content in users_desc:
                c = (content or "").strip()
                if not c or c.strip().lower() == q.lower():
                    continue
                if extract_all_sections(c):
                    continue
                if len(c.split()) <= 6:
                    act_hint = c
                    break
            summary_verb = "summary"
            if "gist" in low_q:
                summary_verb = "gist"
            elif "describe" in low_q or "explain" in low_q:
                summary_verb = "description"
            if act_hint:
                return f"Give {summary_verb} of {sec_typ.title()} {sec_num} of {act_hint}."
            return f"Give {summary_verb} of {sec_typ.title()} {sec_num}."

        return question
    except Exception:
        return question


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _trim_to_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text or "")
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _clean_summary_context(context: str) -> str:
    """Remove common PDF/OCR boilerplate that harms summary quality."""
    if not context:
        return ""

    cleaned_lines = []
    seen = set()
    for raw in (context or "").splitlines():
        line = " ".join(raw.split()).strip()
        if not line:
            continue

        upper = line.upper()
        if (
            upper.startswith("--- PAGE")
            or "PAGE " in upper and " OF " in upper
            or upper.startswith("C/") and "JUDGMENT DATED" in upper
            or upper == "JUDGMENT DATED:"
        ):
            continue

        # Drop exact duplicate lines from OCR tails.
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _dedupe_summary_sentences(text: str) -> str:
    """Remove repeated sentences/segments frequently produced by noisy OCR context."""
    if not text:
        return ""

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    deduped = []
    seen = set()
    for part in parts:
        sentence = " ".join(part.split()).strip()
        if not sentence:
            continue
        key = re.sub(r"[^a-z0-9 ]+", "", sentence.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)

    return " ".join(deduped).strip()


def _sanitize_summary_output(text: str) -> str:
    """
    Remove model self-evaluation artifacts and noisy inline citation markers.
    """
    if not text:
        return ""
    s = text.strip()
    s = re.sub(r"^\s*summary\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\[\d+\]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()

    # Drop common meta-evaluation tails sometimes emitted by instruct models.
    meta_markers = [
        "the summary text is",
        "it accurately captures",
        "adheres to the length requirement",
        "word count",
        "return only the summary text",
    ]
    lower = s.lower()
    cut_idx = -1
    for marker in meta_markers:
        i = lower.find(marker)
        if i != -1 and (cut_idx == -1 or i < cut_idx):
            cut_idx = i
    if cut_idx != -1:
        s = s[:cut_idx].strip()

    return s


def _generate_summary_with_qwen(context: str, question: str, summary_cfg: dict) -> str:
    min_words = summary_cfg.get("min_words")
    max_words = summary_cfg.get("max_words")
    target_words = summary_cfg.get("target_words")

    if min_words is None or max_words is None:
        # Sensible default for summary without explicit size.
        min_words, max_words, target_words = 120, 220, 170

    clean_context = _clean_summary_context(context)
    prompt = f"""
You are a legal summarization assistant.

Task:
- Generate a high-quality summary/description strictly from provided text.
- Keep factual legal details intact (dates, sections, parties, outcomes, court).
- Do not invent facts.
- Ignore OCR/page boilerplate and duplicated headers/footers.

Length rule:
- Output between {min_words} and {max_words} words.
- Target approximately {target_words} words.

User request:
{question}

Document context:
{clean_context}

Return only the summary text.
"""

    # CPU-safe output budget to reduce long waits on local inference.
    max_tokens = max(180, min(500, max_words + 120))
    result = qwen_summary(prompt=prompt, max_tokens=max_tokens, temperature=0.15).strip()
    result = _sanitize_summary_output(result)
    result = _dedupe_summary_sentences(result)
    wc = _word_count(result)

    if wc < min_words or wc > max_words:
        refine_prompt = f"""
Rewrite the following summary to be between {min_words} and {max_words} words
while preserving all key legal facts. Return only rewritten summary.

Summary:
{result}
"""
        result = qwen_summary(prompt=refine_prompt, max_tokens=max_tokens, temperature=0.1).strip()
        result = _sanitize_summary_output(result)
        result = _dedupe_summary_sentences(result)
        wc = _word_count(result)

    if wc > max_words:
        result = _trim_to_words(result, max_words)
    return result


def fast_metadata_router(question: str):
    lower_q = question.lower()

    # Count documents
    # Count documents (including pdf, files, cases)
    if (
        "how many" in lower_q
        and any(word in lower_q for word in ["document", "documents", "pdf", "pdfs", "file", "files", "case", "cases", "judgement", "judgements", "order", "orders"])
    ):
        return "count"


    # Total pages
    if any(x in lower_q for x in ["how many pages", "total pages"]):
        return "total_pages"

    # List documents
    if any(x in lower_q for x in ["what are this documents", "list documents", "show documents"]):
        return "list"

    # Remark: explicit document+pages listing should stay in metadata path.
    if (
        ("document" in lower_q or "documents" in lower_q)
        and ("page" in lower_q or "pages" in lower_q)
        and any(x in lower_q for x in ["list", "according", "with", "along"])
    ):
        return "list_pages"

    # Remark: route consistency checks to metadata engine so semantic RAG
    # does not hallucinate "same court/judge" answers.
    if (
        any(x in lower_q for x in ["same court", "all orders same court", "all documents same court"])
        or ("all" in lower_q and "court" in lower_q and "same" in lower_q)
    ):
        return "same_court"

    if (
        any(x in lower_q for x in ["same judge", "all orders same judge", "all documents same judge"])
        or ("all" in lower_q and "judge" in lower_q and "same" in lower_q)
        or ("all" in lower_q and "justice" in lower_q and "same" in lower_q)
    ):
        return "same_judge"

    return None



def generate_answer(
    question,
    relevant_chunks=None,
    notebook_id=None,
    collection_id=None,
    user_id=None,
    include_global=False,
    global_sub_collection_ids=None,
    specialization=None,
):
    """
    Hybrid Judiciary-Scale RAG Pipeline
    Supports:
    - Metadata routing
    - Single notebook
    - Collection
    - Global master index
    - Hybrid modes
    - Collection-level synthesis
    """

    MAX_TOTAL_CHUNKS = 16
    MAX_TEXT_CHARS = 7000
    allowed_global_ids = set(_normalize_collection_filters(global_sub_collection_ids))
    
    loaded_global = None
    question = _maybe_expand_section_followup_from_history(
        question,
        collection_id=collection_id,
        notebook_id=notebook_id,
        user_id=user_id,
    )
    act_mentions = _normalize_act_mentions(extract_acts_from_question(question))
    act_override_active = bool(act_mentions)
    effective_allowed_global_ids = set() if act_override_active else set(allowed_global_ids)
    llm_question = (
        f"{_specialization_instructions(specialization)}"
        f"{_procedural_guidance_instruction(question)}"
        f"{_ni_act_guidance_instruction(question, act_mentions, specialization)}"
        f"{question}"
    )
    summary_cfg = _detect_summary_request(question)

    print("\n============================")
    print("📨 QUESTION:", question)
    print("============================")

    # ==================================================
    # 0️⃣ FAST METADATA ROUTING (NO LLM)
    # ==================================================

    metadata_operation = fast_metadata_router(question)

    if metadata_operation:
        from rag.metadata_engine import handle_metadata_intent

        metadata_answer = handle_metadata_intent(
            intent={
                "intent_type": "metadata",
                "operation": metadata_operation,
                "entities": {},
                "filters": {}
            },
            collection_id=collection_id,
            user_id=user_id,
            notebook_id=notebook_id,
        )
        return _make_response(str(metadata_answer), citations=[], mode="metadata")

    # ==================================================
    # 0.5️⃣ GENERIC METADATA QA FALLBACK (NON-HARDCODED)
    # ==================================================
    # Remark: avoid adding one-off rules for every metadata phrasing.
    try:
        from rag.metadata_engine import answer_from_metadata_context
        generic_metadata_answer = answer_from_metadata_context(
            question=question,
            collection_id=collection_id,
            user_id=user_id,
            notebook_id=notebook_id,
        )
        if generic_metadata_answer:
            lower_ans = generic_metadata_answer.lower()
            # If metadata cannot answer, continue to semantic retrieval.
            if "not determinable from metadata" not in lower_ans:
                return _make_response(str(generic_metadata_answer), citations=[], mode="metadata-generic")
    except Exception:
        pass




    # ==================================================
    # 0.75️⃣ SECTION-FIRST RETRIEVAL (FOR LEGAL ACTS)
    # ==================================================
    # Try to match sections before doing semantic search
    
    print(f"🔍 Attempting section-first retrieval...")
    section_chunks, section_strategy = retrieve_by_section_first(
        question=question,
        notebook_id=notebook_id,
        collection_id=collection_id,
        user_id=user_id,
    )
    if include_global and effective_allowed_global_ids and not notebook_id and not collection_id:
        section_chunks = _filter_section_hits_by_global_subcollections(
            section_chunks,
            list(effective_allowed_global_ids),
        )

    if section_strategy == "section_ambiguous_missing_act":
        return _make_response(
            "Multiple Acts contain that section number. Please mention the Act name (for example: CPC, CrPC/BNSS, Evidence Act) so I can fetch the correct section.",
            citations=[],
            mode="section_ambiguous",
        )
    if section_strategy == "section_act_not_found_in_scope":
        return _make_response(
            "I could not find that Act in the current collection scope. Please specify the exact book/Act name available in this collection.",
            citations=[],
            mode="section_act_missing",
        )
    if section_strategy == "section_book_not_found_in_scope":
        return _make_response(
            "I could not find that reference document/book in this collection scope. Please choose a book from the collection list and ask again.",
            citations=[],
            mode="section_book_missing",
        )
    if section_strategy == "section_ambiguous_missing_book":
        book_names = []
        seen = set()
        for h in section_chunks[:10]:
            name = (h.get("filename") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            book_names.append(name)
        suffix = ""
        if book_names:
            suffix = " Matching books: " + "; ".join(book_names[:6]) + "."
        return _make_response(
            "This section number appears in multiple books in this collection. Please specify the book name so I can fetch the correct text." + suffix,
            citations=[],
            mode="section_ambiguous_book",
        )
    
    if section_chunks and section_strategy == "section_exact_match":
        # If section-exact results span multiple notebooks and user did not
        # specify Act/book, ask for notebook disambiguation first.
        section_nb_ids = [c.get("notebook_id") for c in section_chunks if c.get("notebook_id")]
        unique_section_nbs = list(dict.fromkeys(section_nb_ids))
        acts_in_q_for_section = extract_acts_from_question(question)
        if len(unique_section_nbs) > 1 and not acts_in_q_for_section:
            name_map = _lookup_notebook_filenames(unique_section_nbs[:8], user_id=user_id)
            options = [name_map.get(nb) for nb in unique_section_nbs[:6] if name_map.get(nb)]
            hint = f" Candidate notebooks: {'; '.join(options)}." if options else ""
            return _make_response(
                "This section appears in multiple notebooks. Please specify the reference document/book name." + hint,
                citations=[],
                mode="section_ambiguous_book",
            )

        print(f"✨ Found exact section match! Using section-first result")
        # Convert to scored format and return immediately
        collected_chunks = [chunk["text"] for chunk in section_chunks]
        context = "\n\n".join(collected_chunks)
        
        if len(context) > 7000:
            context = context[:7000]
        
        if summary_cfg.get("is_summary"):
            try:
                answer = _generate_summary_with_qwen(
                    context=context,
                    question=llm_question,
                    summary_cfg=summary_cfg,
                )
            except Exception:
                answer = _build_section_exact_answer(
                    question,
                    section_chunks,
                    llm_question=llm_question,
                )
        else:
            answer = _build_section_exact_answer(
                question,
                section_chunks,
                llm_question=llm_question,
            )
        citations = [
            {
                "source_type": "section_exact",
                "section_id": chunk.get("section_id"),
                "notebook_id": chunk.get("notebook_id"),
                "chunk_index": chunk.get("chunk_index"),
                "score": 1.0,
                "preview": chunk["text"][:180],
            }
            for chunk in section_chunks[:3]
        ]
        citations = _enrich_citation_filenames(citations, user_id=user_id)
        return _make_response(answer, citations=citations, mode="section_exact")

    # Guardrail: in collection mode, section-only queries without act/book mention
    # should ask for book disambiguation instead of drifting into semantic mixing.
    if collection_id and not notebook_id:
        try:
            sections_in_q = extract_all_sections(question)
            acts_in_q = extract_acts_from_question(question)
            if sections_in_q and not acts_in_q:
                from db import get_repo
                repo = get_repo()
                cur = repo.conn.cursor()
                cur.execute(
                    """
                    SELECT c.is_global
                    FROM collections c
                    WHERE c.collection_id = %s
                    """,
                    (collection_id,),
                )
                row = cur.fetchone()
                is_global_collection = bool(row and row[0])
                if is_global_collection:
                    cur.execute(
                        """
                        SELECT COUNT(DISTINCT dm.document_id)
                        FROM document_metadata dm
                        WHERE dm.collection_id = %s
                          AND dm.document_role = 'ReferenceBook'
                        """,
                        (collection_id,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT COUNT(DISTINCT dm.document_id)
                        FROM document_metadata dm
                        WHERE dm.collection_id = %s
                          AND dm.user_id = %s
                          AND dm.document_role = 'ReferenceBook'
                        """,
                        (collection_id, user_id),
                    )
                ref_count = int((cur.fetchone() or [0])[0] or 0)
                if ref_count > 1:
                    cur.execute(
                        """
                        SELECT dm.filename
                        FROM document_metadata dm
                        WHERE dm.collection_id = %s
                          AND dm.document_role = 'ReferenceBook'
                          AND (%s::boolean = TRUE OR dm.user_id = %s)
                        ORDER BY dm.created_at DESC NULLS LAST
                        LIMIT 6
                        """,
                        (collection_id, is_global_collection, user_id),
                    )
                    names = [r[0] for r in cur.fetchall() if r and r[0]]
                    suffix = f" Examples: {'; '.join(names)}." if names else ""
                    return _make_response(
                        "Multiple books in this collection can contain that section number. Please specify the book/Act name." + suffix,
                        citations=[],
                        mode="section_ambiguous_book",
                    )
        except Exception:
            pass


    # ==================================================
    # 1️⃣ ROUTER
    # ==================================================

    router_decision = classify_question(question)
    print("🧭 ROUTER DECISION:", router_decision)

    # Act-first retrieval override:
    # If query explicitly mentions an Act, constrain retrieval to matching docs.
    act_scoped_doc_ids = set()
    if act_override_active:
        for act in act_mentions:
            act_scoped_doc_ids.update(
                find_documents_with_act(
                    collection_id=collection_id,
                    user_id=user_id,
                    act_name=act,
                )
            )
        print(f"🎯 Act-first override active: acts={act_mentions}, docs={len(act_scoped_doc_ids)}")


    # ==================================================
    # 2️⃣ EMBED QUESTION
    # ==================================================

    q_emb = embed_texts([question])[0]
    q = np.array([q_emb], dtype="float32")

    # ==================================================
    # 3️⃣ SINGLE NOTEBOOK MODE
    # ==================================================

    if notebook_id and not collection_id:
        scored_chunks = []

        allow_notebook_scope = (not act_override_active) or _document_has_any_act(notebook_id, act_mentions)

        loaded = load_vectors(notebook_id) if allow_notebook_scope else None
        if loaded:
            index, metadata = loaded
            distances, indices = index.search(q, 10)

            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                _append_scored_chunk(scored_chunks, float(dist), metadata[idx], "notebook")

        if include_global:
            loaded_global = load_global_index()

        if loaded_global:
            global_index, global_metadata = loaded_global
            distances, indices = global_index.search(q, 15)

            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1:
                    meta = global_metadata[idx]
                    if not _allow_global_chunk(meta, effective_allowed_global_ids):
                        continue
                    if act_override_active:
                        nb = str(meta.get("notebook_id") or "").strip()
                        if not nb or (act_scoped_doc_ids and nb not in act_scoped_doc_ids):
                            continue
                    _append_scored_chunk(scored_chunks, float(dist), meta, "global")

        if act_override_active and not scored_chunks:
            return _make_response(
                "I could not find that Act in the current scope. Please verify Act name or enable relevant global collections.",
                citations=[],
                mode="act_scoped_not_found",
            )

        if not scored_chunks:
            return _make_response("Not mentioned in the documents.", citations=[], mode="notebook")

        selected_chunks = _select_top_unique_chunks(scored_chunks, MAX_TOTAL_CHUNKS)
        collected_chunks = [chunk["text"] for chunk in selected_chunks]

        # -----------------------------------------
        # Add metadata intelligence (if available)
        # -----------------------------------------

        metadata_summary = ""

        try:
            from db import get_repo
            repo = get_repo()
            cur = repo.conn.cursor()

            cur.execute("""
                SELECT case_number, order_date, document_type
                FROM document_metadata
                WHERE document_id = %s
            """, (notebook_id,))

            meta = cur.fetchone()

            if meta:
                metadata_summary = f"""
        DOCUMENT METADATA:
        Case Number: {meta[0] or "N/A"}
        Order Date: {meta[1] or "N/A"}
        Document Type: {meta[2] or "Unknown"}
        """
        except Exception:
            metadata_summary = ""

        # -----------------------------------------
        # Build context
        # -----------------------------------------

        context = metadata_summary + "\n\n" + "\n\n".join(collected_chunks)


        if len(context) > MAX_TEXT_CHARS:
            context = context[:MAX_TEXT_CHARS]

        if summary_cfg.get("is_summary"):
            try:
                answer = _generate_summary_with_qwen(
                    context=context,
                    question=llm_question,
                    summary_cfg=summary_cfg,
                )
            except Exception:
                answer = ask_llm(context, llm_question)
        else:
            answer = ask_llm(context, llm_question)
        citations = _build_citations(selected_chunks, max_citations=4)
        citations = _enrich_citation_filenames(citations, user_id=user_id)
        return _make_response(answer, citations=citations, mode="notebook")

    # ==================================================
    # 4️⃣ COLLECTION MODE (WITH SYNTHESIS)
    # ==================================================

    if collection_id and user_id:

        notebook_ids = get_collection_notebooks(collection_id, user_id)
        if act_override_active:
            notebook_ids = [nb for nb in notebook_ids if nb in act_scoped_doc_ids]

        scored_chunks = []
        print(f"📚 Collection mode — Notebooks found: {len(notebook_ids)}")
        per_notebook_k = 2 if len(notebook_ids) > 20 else 3

        for nb_id in notebook_ids:
            loaded = load_vectors(nb_id)
            if not loaded:
                continue

            index, metadata = loaded
            distances, indices = index.search(q, per_notebook_k)

            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                _append_scored_chunk(scored_chunks, float(dist), metadata[idx], "collection")

        
        

        # ---- Add global intelligence ----
        if include_global:
            loaded_global = load_global_index()
        if loaded_global:
            global_index, global_metadata = loaded_global
            distances, indices = global_index.search(q, 15)

            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                meta = global_metadata[idx]
                if not _allow_global_chunk(meta, effective_allowed_global_ids):
                    continue
                if act_override_active:
                    nb = str(meta.get("notebook_id") or "").strip()
                    if not nb or (act_scoped_doc_ids and nb not in act_scoped_doc_ids):
                        continue
                _append_scored_chunk(scored_chunks, float(dist), meta, "global")

        # ==================================================
        # 5️⃣ SYNTHESIS LOGIC
        # ==================================================

        print(f"🧠 Total chunks collected: {len(scored_chunks)}")
        if act_override_active and not scored_chunks:
            return _make_response(
                "I could not find that Act in the current collection scope. Please verify Act name or include relevant source.",
                citations=[],
                mode="act_scoped_not_found",
            )
        if not scored_chunks:
            return _make_response("Not mentioned in the documents.", citations=[], mode="collection")

        selected_chunks = _select_top_unique_chunks(scored_chunks, MAX_TOTAL_CHUNKS)
        maybe_clarify = _maybe_return_collection_clarification(
            question=question,
            selected_chunks=selected_chunks,
            user_id=user_id,
        )
        if maybe_clarify:
            return maybe_clarify
        all_chunks = [chunk["text"] for chunk in selected_chunks]

        context = "\n\n".join(all_chunks)

        if len(context) > MAX_TEXT_CHARS:
            context = context[:MAX_TEXT_CHARS]

        if summary_cfg.get("is_summary"):
            try:
                answer = _generate_summary_with_qwen(
                    context=context,
                    question=llm_question,
                    summary_cfg=summary_cfg,
                )
            except Exception:
                answer = ask_llm(context, llm_question)
        else:
            answer = ask_llm(context, llm_question)
        citations = _build_citations(selected_chunks, max_citations=3)
        citations = _enrich_citation_filenames(citations, user_id=user_id)
        return _make_response(answer, citations=citations, mode="collection")


    # ==================================================
    # 6️⃣ GLOBAL ONLY MODE
    # ==================================================

    if include_global:
        loaded_global = load_global_index()

        if loaded_global:
            global_index, global_metadata = loaded_global
            distances, indices = global_index.search(q, 20)

            scored_chunks = []

            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                meta = global_metadata[idx]
                if not _allow_global_chunk(meta, effective_allowed_global_ids):
                    continue
                if act_override_active:
                    nb = str(meta.get("notebook_id") or "").strip()
                    if not nb or (act_scoped_doc_ids and nb not in act_scoped_doc_ids):
                        continue
                _append_scored_chunk(scored_chunks, float(dist), meta, "global")

            if act_override_active and not scored_chunks:
                return _make_response(
                    "I could not find that Act in the current scope. Please verify Act name or choose relevant source collection.",
                    citations=[],
                    mode="act_scoped_not_found",
                )

            if scored_chunks:
                selected_chunks = _select_top_unique_chunks(scored_chunks, MAX_TOTAL_CHUNKS)
                maybe_clarify = _maybe_return_collection_clarification(
                    question=question,
                    selected_chunks=selected_chunks,
                    user_id=user_id,
                )
                if maybe_clarify:
                    return maybe_clarify
                chunks = [chunk["text"] for chunk in selected_chunks]
                context = "\n\n".join(chunks)

                if len(context) > MAX_TEXT_CHARS:
                    context = context[:MAX_TEXT_CHARS]

                if summary_cfg.get("is_summary"):
                    try:
                        answer = _generate_summary_with_qwen(
                            context=context,
                            question=llm_question,
                            summary_cfg=summary_cfg,
                        )
                    except Exception:
                        answer = ask_llm(context, llm_question)
                else:
                    answer = ask_llm(context, llm_question)
                citations = _build_citations(selected_chunks, max_citations=3)
                citations = _enrich_citation_filenames(citations, user_id=user_id)
                return _make_response(answer, citations=citations, mode="global")

    return _make_response("Not mentioned in the documents.", citations=[], mode="none")
