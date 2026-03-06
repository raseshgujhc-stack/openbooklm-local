"""
Main FastAPI backend for OpenBookLM local deployment.

Responsibilities:
- Authentication and admin APIs
- Upload/job orchestration
- Chat submission/polling APIs
- Collection/notebook management
- Runtime schema safety checks on startup
"""

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    Header,
    Request,
)
import threading
from ingest_worker import ingest_worker_loop
from rag.metadata_retry_worker import metadata_retry_worker_loop
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import bcrypt
import threading
import json
import re
import time
import hashlib
from pathlib import Path
import os
from fastapi.responses import StreamingResponse
from typing import Optional, List
from psycopg2.extras import Json

from db import get_repo   # ✅ PostgreSQL repo

from rag.pdf_reader import read_pdf
from rag.text_splitter import split_text
from rag.embedder import embed
from rag.rag_pipeline import generate_answer
from rag.vector_store import (
    save_vectors,
    load_vectors,
    delete_vectors,
    load_texts,
    get_collection_notebooks,
)
from rag.podcast_worker_qwen import run_podcast_job_qwen, synthesize_podcast_audio
from rag.ingest import ingest_document
from rag.clear import hard_delete_notebook
from rag.act_catalog import normalize_act_name as catalog_normalize_act_name
from rag.chunker import extract_all_sections
from rag.script_rectifier import normalize_and_validate_podcast_script

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
PODCAST_RUNNING_TIMEOUT_SECONDS = int(os.getenv("PODCAST_RUNNING_TIMEOUT_SECONDS", "1200"))

# --------------------------------------------------
# App init
# --------------------------------------------------

app = FastAPI(title="Local NotebookLM Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    # Remark: lightweight liveness for UI service-status checks.
    return {"status": "ok", "service": "backend"}


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "microphone=(self)"
    response.headers["Cache-Control"] = "no-store"
    return response

def ensure_collection_chat_history_schema():
    """
    Ensure collection chat history table/indexes exist in PostgreSQL.
    This avoids silent history loss when migrations were skipped.
    """
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS collection_chat_history (
            id SERIAL PRIMARY KEY,
            collection_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_collection_chat_collection ON collection_chat_history(collection_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_collection_chat_user ON collection_chat_history(user_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_collection_chat_created ON collection_chat_history(created_at)"
    )
    repo.conn.commit()

def ensure_chat_archive_schema():
    """
    Persist cleared chat sessions so users can view archived history.
    """
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history_archive (
            id BIGSERIAL PRIMARY KEY,
            notebook_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP,
            archived_at TIMESTAMP NOT NULL DEFAULT NOW(),
            archived_by TEXT NOT NULL,
            archive_session_id TEXT
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_history_archive_nb ON chat_history_archive(notebook_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_history_archive_user ON chat_history_archive(archived_by)"
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS collection_chat_history_archive (
            id BIGSERIAL PRIMARY KEY,
            collection_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP,
            archived_at TIMESTAMP NOT NULL DEFAULT NOW(),
            archived_by TEXT NOT NULL,
            archive_session_id TEXT
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_collection_chat_archive_col ON collection_chat_history_archive(collection_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_collection_chat_archive_user ON collection_chat_history_archive(user_id)"
    )
    cur.execute(
        "ALTER TABLE chat_history_archive ADD COLUMN IF NOT EXISTS archive_session_id TEXT"
    )
    cur.execute(
        "ALTER TABLE collection_chat_history_archive ADD COLUMN IF NOT EXISTS archive_session_id TEXT"
    )
    repo.conn.commit()

def ensure_chat_jobs_schema():
    """Ensure async chat job table exists."""
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_jobs (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            notebook_id TEXT,
            collection_id TEXT,
            question TEXT NOT NULL,
            include_global BOOLEAN DEFAULT FALSE,
            global_sub_collection_ids JSONB DEFAULT '[]'::jsonb,
            specialization TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            result TEXT,
            result_payload JSONB,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute("ALTER TABLE chat_jobs ADD COLUMN IF NOT EXISTS result_payload JSONB")
    cur.execute("ALTER TABLE chat_jobs ADD COLUMN IF NOT EXISTS global_sub_collection_ids JSONB DEFAULT '[]'::jsonb")
    cur.execute("ALTER TABLE chat_jobs ADD COLUMN IF NOT EXISTS specialization TEXT")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_jobs_user ON chat_jobs(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_jobs_status ON chat_jobs(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_jobs_created ON chat_jobs(created_at)")
    repo.conn.commit()

def ensure_chat_answer_cache_schema():
    """Cache repeated chat answers to reduce latency/cost for duplicate questions."""
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_answer_cache (
            id BIGSERIAL PRIMARY KEY,
            scope_type TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            user_scope TEXT,
            include_global BOOLEAN NOT NULL DEFAULT FALSE,
            question_hash TEXT NOT NULL,
            question_norm TEXT NOT NULL,
            answer_text TEXT NOT NULL,
            answer_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            hit_count INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
            UNIQUE (scope_type, scope_id, user_scope, include_global, question_hash)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chat_answer_cache_lookup
        ON chat_answer_cache(scope_type, scope_id, user_scope, include_global, question_hash)
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_answer_cache_updated ON chat_answer_cache(updated_at)"
    )
    repo.conn.commit()


def ensure_specialization_profiles_schema():
    """
    Store specialization prompts/knowledge in PostgreSQL for centralized updates.
    """
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS specialization_profiles (
            key TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            config JSONB NOT NULL DEFAULT '{}'::jsonb,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_specialization_profiles_active ON specialization_profiles(is_active)"
    )

    seeds = [
        (
            "general",
            "General",
            {
                "instruction": "Answer with high legal precision and clear structure.",
            },
        ),
        (
            "criminal",
            "Criminal Law",
            {
                "instruction": "Focus on criminal law procedure, ingredients of offence, burden of proof, and judicial tests.",
            },
        ),
        (
            "civil",
            "Civil Law",
            {
                "instruction": "Focus on civil remedies, pleadings, limitation, maintainability, and decree/enforcement implications.",
            },
        ),
        (
            "constitutional",
            "Constitutional Law",
            {
                "instruction": "Focus on constitutional principles, fundamental rights, proportionality, and precedent hierarchy.",
            },
        ),
        (
            "evidence",
            "Evidence Law",
            {
                "instruction": "Focus on admissibility, relevancy, presumptions, burden shifts, and evidentiary value.",
            },
        ),
        (
            "procedural",
            "Procedural Law",
            {
                "instruction": "Focus on procedural compliance, timelines, jurisdiction, and defects curable/incurable.",
            },
        ),
        (
            "tax",
            "Tax Law",
            {
                "instruction": "Focus on charging provisions, exemptions, classification, and ratio of cited tax precedents.",
            },
        ),
        (
            "ni_act",
            "Negotiable Instruments Act",
            {
                "instruction": "Focus on cheque dishonour litigation under NI Act with stage-wise risk analysis.",
                "act_aliases": [
                    "Negotiable Instruments Act",
                    "NI Act",
                    "N.I. Act",
                ],
                "stages": [
                    "Cheque issuance and presentation within limitation",
                    "Dishonour memo from bank",
                    "Statutory demand notice within prescribed period",
                    "Payment window after notice",
                    "Complaint filing before Magistrate within limitation",
                    "Summoning and plea",
                    "Complainant evidence and cross-examination",
                    "Accused defence evidence",
                    "Final arguments and judgment",
                    "Compensation/sentence and appeal/revision",
                ],
                "presumptions": [
                    "Section 118(a): presumption as to consideration",
                    "Section 139: presumption that cheque was issued towards legally enforceable debt/liability",
                    "Presumptions are rebuttable on preponderance of probabilities",
                ],
                "missing_info_checklist": [
                    "Cheque number/date/amount and bank details",
                    "Date of presentation and date of dishonour memo",
                    "Dishonour reason in bank return memo",
                    "Demand notice date, mode, service proof",
                    "Notice receipt/refusal tracking details",
                    "Complaint filing date and jurisdiction facts",
                    "Proof of underlying legally enforceable liability",
                ],
            },
        ),
    ]
    for key, title, config in seeds:
        cur.execute(
            """
            INSERT INTO specialization_profiles (key, title, config, is_active, updated_at)
            VALUES (%s, %s, %s, TRUE, NOW())
            ON CONFLICT (key) DO UPDATE
            SET title = EXCLUDED.title,
                config = specialization_profiles.config || EXCLUDED.config,
                is_active = TRUE,
                updated_at = NOW()
            """,
            (key, title, Json(config)),
        )

    repo.conn.commit()


def normalize_answer_payload(answer_result):
    """Normalize legacy string and new structured RAG responses."""
    if isinstance(answer_result, dict):
        answer_text = str(answer_result.get("answer", "")).strip()
        return (answer_text or "No answer"), answer_result
    answer_text = str(answer_result or "").strip()
    normalized_text = answer_text or "No answer"
    return normalized_text, {"answer": normalized_text, "citations": [], "mode": "legacy"}

def _normalize_question_for_cache(question: str) -> str:
    normalized = (question or "").strip().lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def _question_hash(normalized_question: str) -> str:
    return hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()

def _is_recheck_request(question: str) -> bool:
    q = (question or "").lower()
    tokens = (
        "recheck",
        "check again",
        "verify again",
        "regenerate",
        "re-generate",
        "fresh answer",
        "again with",
        "re-evaluate",
        "reevaluate",
    )
    return any(token in q for token in tokens)

def _needs_contextual_rewrite(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    low = q.lower()
    if len(q.split()) <= 5:
        return True
    if re.search(r"\b(this|that|it|they|those|these|same|above|previous|earlier)\b", low):
        return True
    if re.match(r"^(and|also|then|so|what about|why|how about)\b", low):
        return True
    return False

def _fetch_recent_turns(
    cur,
    *,
    user_id: str,
    collection_id: Optional[str] = None,
    notebook_id: Optional[str] = None,
    limit_rows: int = 10,
):
    if collection_id:
        cur.execute(
            """
            SELECT role, content
            FROM collection_chat_history
            WHERE collection_id = %s AND user_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (collection_id, user_id, limit_rows),
        )
    elif notebook_id:
        cur.execute(
            """
            SELECT role, content
            FROM chat_history
            WHERE notebook_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (notebook_id, limit_rows),
        )
    else:
        return []
    rows = cur.fetchall() or []
    rows.reverse()
    return rows

def _expand_question_with_context(
    cur,
    *,
    question: str,
    user_id: str,
    collection_id: Optional[str] = None,
    notebook_id: Optional[str] = None,
) -> str:
    """
    Generic context-aware rewrite for short/abstract follow-ups across chats.
    Keeps retrieval deterministic by embedding a compact prior-turn context.
    """
    if not _needs_contextual_rewrite(question):
        return question

    try:
        rows = _fetch_recent_turns(
            cur,
            user_id=user_id,
            collection_id=collection_id,
            notebook_id=notebook_id,
            limit_rows=8,
        )
        if not rows:
            return question

        compact = []
        for role, content in rows[-6:]:
            text = " ".join((content or "").split())
            if not text:
                continue
            text = text[:220]
            prefix = "User" if (role or "").lower() == "user" else "Assistant"
            compact.append(f"{prefix}: {text}")
        if not compact:
            return question

        return (
            "Use recent conversation context to interpret the question.\n"
            f"Conversation:\n{chr(10).join(compact)}\n"
            f"Current question: {question}"
        )
    except Exception:
        return question

def _get_collection_is_global(cur, collection_id: Optional[str]) -> bool:
    if not collection_id:
        return False
    cur.execute(
        "SELECT is_global FROM collections WHERE collection_id = %s",
        (collection_id,),
    )
    row = cur.fetchone()
    return bool(row and row[0])


def _normalize_global_sub_collection_ids(cur, ids: Optional[List[str]]) -> List[str]:
    if not ids:
        return []

    cleaned = []
    seen = set()
    for raw in ids:
        cid = str(raw or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        cleaned.append(cid)

    if not cleaned:
        return []

    placeholders = ",".join(["%s"] * len(cleaned))
    cur.execute(
        f"""
        SELECT collection_id
        FROM collections
        WHERE is_global = TRUE
          AND collection_id IN ({placeholders})
        """,
        tuple(cleaned),
    )
    valid = {row[0] for row in (cur.fetchall() or [])}
    return [cid for cid in cleaned if cid in valid]

def _resolve_cache_scope(
    cur,
    *,
    user_id: str,
    notebook_id: Optional[str] = None,
    collection_id: Optional[str] = None,
) -> tuple[str, str, Optional[str]]:
    if collection_id:
        is_global_collection = _get_collection_is_global(cur, collection_id)
        return ("collection", collection_id, None if is_global_collection else user_id)
    return ("notebook", notebook_id or "", user_id)

def _lookup_cached_answer(
    cur,
    *,
    scope_type: str,
    scope_id: str,
    user_scope: Optional[str],
    include_global: bool,
    question_hash: str,
):
    cur.execute(
        """
        SELECT answer_text, answer_payload
        FROM chat_answer_cache
        WHERE scope_type = %s
          AND scope_id = %s
          AND user_scope IS NOT DISTINCT FROM %s
          AND include_global = %s
          AND question_hash = %s
        """,
        (scope_type, scope_id, user_scope, bool(include_global), question_hash),
    )
    return cur.fetchone()

def _mark_cached_answer_hit(
    cur,
    *,
    scope_type: str,
    scope_id: str,
    user_scope: Optional[str],
    include_global: bool,
    question_hash: str,
):
    cur.execute(
        """
        UPDATE chat_answer_cache
        SET hit_count = hit_count + 1, updated_at = NOW()
        WHERE scope_type = %s
          AND scope_id = %s
          AND user_scope IS NOT DISTINCT FROM %s
          AND include_global = %s
          AND question_hash = %s
        """,
        (scope_type, scope_id, user_scope, bool(include_global), question_hash),
    )

def _upsert_cached_answer(
    cur,
    *,
    scope_type: str,
    scope_id: str,
    user_scope: Optional[str],
    include_global: bool,
    question_hash: str,
    question_norm: str,
    answer_text: str,
    answer_payload: dict,
):
    cur.execute(
        """
        INSERT INTO chat_answer_cache (
            scope_type, scope_id, user_scope, include_global,
            question_hash, question_norm, answer_text, answer_payload,
            hit_count, created_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1, NOW(), NOW())
        ON CONFLICT (scope_type, scope_id, user_scope, include_global, question_hash)
        DO UPDATE
        SET answer_text = EXCLUDED.answer_text,
            answer_payload = EXCLUDED.answer_payload,
            hit_count = chat_answer_cache.hit_count + 1,
            updated_at = NOW()
        """,
        (
            scope_type,
            scope_id,
            user_scope,
            bool(include_global),
            question_hash,
            question_norm,
            answer_text,
            Json(answer_payload or {}),
        ),
    )

def _answer_with_cache(
    cur,
    *,
    question: str,
    user_id: str,
    notebook_id: Optional[str] = None,
    collection_id: Optional[str] = None,
    include_global: bool = False,
    global_sub_collection_ids: Optional[List[str]] = None,
    specialization: Optional[str] = None,
    relevant_chunks=None,
):
    total_start = time.perf_counter()
    cache_lookup_ms = 0.0
    generation_ms = 0.0
    used_cache = False

    # Step 1: resolve act-only follow-up (e.g., "BSA" after "Section 21?")
    original_question = question
    question = _expand_followup_act_question(
        cur,
        question=question,
        user_id=user_id,
        collection_id=collection_id,
        notebook_id=notebook_id,
    )
    # Step 2: generic context-aware rewrite for abstract/short follow-ups.
    # If step 1 already produced an explicit disambiguated question, do not
    # wrap context again (prevents ambiguity loops).
    if question == original_question:
        question = _expand_question_with_context(
            cur,
            question=question,
            user_id=user_id,
            collection_id=collection_id,
            notebook_id=notebook_id,
        )

    # Scope controls whether cache is user-private or shared (global collections).
    scope_type, scope_id, user_scope = _resolve_cache_scope(
        cur,
        user_id=user_id,
        notebook_id=notebook_id,
        collection_id=collection_id,
    )

    filter_key = ",".join(sorted(global_sub_collection_ids or []))
    spec_key = (specialization or "").strip().lower()
    cacheable_question = f"{question}\n\n[[GLOBAL_FILTERS:{filter_key}]]\n[[SPECIALIZATION:{spec_key}]]"
    question_norm = _normalize_question_for_cache(cacheable_question)
    qhash = _question_hash(question_norm)
    force_regen = _is_recheck_request(question)

    cached_row = None
    # Explicit recheck-style prompts bypass cache and force fresh generation.
    if not force_regen and question_norm:
        cache_start = time.perf_counter()
        cached_row = _lookup_cached_answer(
            cur,
            scope_type=scope_type,
            scope_id=scope_id,
            user_scope=user_scope,
            include_global=include_global,
            question_hash=qhash,
        )
        cache_lookup_ms = (time.perf_counter() - cache_start) * 1000.0

    if cached_row:
        used_cache = True
        _mark_cached_answer_hit(
            cur,
            scope_type=scope_type,
            scope_id=scope_id,
            user_scope=user_scope,
            include_global=include_global,
            question_hash=qhash,
        )
        answer_text = str(cached_row[0] or "No answer").strip() or "No answer"
        payload = cached_row[1] or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}
        answer_payload = payload if isinstance(payload, dict) else {}
        answer_payload["answer"] = answer_payload.get("answer") or answer_text
    else:
        gen_start = time.perf_counter()
        if collection_id:
            answer_result = generate_answer(
                question=question,
                collection_id=collection_id,
                user_id=user_id,
                include_global=bool(include_global),
                global_sub_collection_ids=global_sub_collection_ids or [],
                specialization=specialization,
            )
        else:
            answer_result = generate_answer(
                question=question,
                relevant_chunks=relevant_chunks or [],
                notebook_id=notebook_id,
                user_id=user_id,
                include_global=bool(include_global),
                global_sub_collection_ids=global_sub_collection_ids or [],
                specialization=specialization,
            )
        generation_ms = (time.perf_counter() - gen_start) * 1000.0
        answer_text, answer_payload = normalize_answer_payload(answer_result)

        if question_norm:
            _upsert_cached_answer(
                cur,
                scope_type=scope_type,
                scope_id=scope_id,
                user_scope=user_scope,
                include_global=include_global,
                question_hash=qhash,
                question_norm=question_norm,
                answer_text=answer_text,
                answer_payload=answer_payload,
            )

    # Runtime metrics are surfaced in UI for latency transparency.
    total_ms = (time.perf_counter() - total_start) * 1000.0
    runtime = {
        "cached": used_cache,
        "cache_lookup_ms": round(cache_lookup_ms, 2),
        "generation_ms": round(generation_ms, 2),
        "total_ms": round(total_ms, 2),
        "cache_scope": scope_type,
        "shared_cache": user_scope is None,
    }
    answer_payload["runtime"] = runtime
    return answer_text, answer_payload


def _expand_followup_act_question(
    cur,
    *,
    question: str,
    user_id: str,
    collection_id: Optional[str] = None,
    notebook_id: Optional[str] = None,
) -> str:
    """
    Carry conversational context for section disambiguation:
    "Section 21?" -> "BSA" becomes "What is Section 21 of <BSA canonical>?"
    """
    q = (question or "").strip()
    if not q:
        return question

    # Skip if question already contains section/article markers.
    if re.search(r"\b(section|sections|article|articles|chapter|part)\b", q, flags=re.IGNORECASE):
        return question

    canonical_act = catalog_normalize_act_name(q)
    # Keep raw user phrase as fallback when catalog normalization misses.
    act_or_book_phrase = canonical_act or q

    # Keep this only for short act-only follow-ups.
    if len(q.split()) > 6 or len(q) > 80:
        return question

    try:
        if collection_id:
            cur.execute(
                """
                SELECT role, content
                FROM collection_chat_history
                WHERE collection_id = %s AND user_id = %s
                ORDER BY id DESC
                LIMIT 12
                """,
                (collection_id, user_id),
            )
        elif notebook_id:
            cur.execute(
                """
                SELECT role, content
                FROM chat_history
                WHERE notebook_id = %s
                ORDER BY id DESC
                LIMIT 12
                """,
                (notebook_id,),
            )
        else:
            return question

        rows = cur.fetchall() or []
        if not rows:
            return question

        # Only trigger rewrite when previous assistant explicitly asked for
        # disambiguation (act or book/notebook).
        latest_assistant = next((r[1] for r in rows if (r[0] or "").lower() == "assistant"), "") or ""
        assistant_low = latest_assistant.lower()
        disambiguation_markers = [
            "multiple acts contain that section number",
            "multiple books in this collection can contain that section number",
            "this section appears in multiple notebooks",
            "please specify the reference document/book name",
            "please specify the book name",
        ]
        if not any(m in assistant_low for m in disambiguation_markers):
            return question

        latest_user_with_section = None
        for role, content in rows:
            if (role or "").lower() != "user":
                continue
            secs = extract_all_sections(content or "")
            if secs:
                latest_user_with_section = (content or "", secs)
                break

        if not latest_user_with_section:
            return question

        _, secs = latest_user_with_section
        if len(secs) == 1:
            num, typ = secs[0]
            return f"What is {typ.title()} {num} of {act_or_book_phrase}?"

        nums = [num for num, _ in secs[:4]]
        joined = ", ".join(nums[:-1]) + (f" and {nums[-1]}" if len(nums) > 1 else nums[0])
        return f"What are Sections {joined} of {act_or_book_phrase}?"
    except Exception:
        return question


def ensure_metadata_retry_schema():
    """
    Ensure extraction status fields and retry queue exist.
    """
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        "ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS extraction_status TEXT DEFAULT 'complete'"
    )
    cur.execute(
        "ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS needs_review BOOLEAN DEFAULT FALSE"
    )
    cur.execute(
        "ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0"
    )
    cur.execute(
        "ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS last_retry_at TIMESTAMP NULL"
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata_retry_jobs (
            id SERIAL PRIMARY KEY,
            document_id TEXT NOT NULL UNIQUE,
            user_id TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            attempts INTEGER NOT NULL DEFAULT 0,
            next_retry_at TIMESTAMP NOT NULL DEFAULT NOW(),
            last_error TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_metadata_retry_jobs_status ON metadata_retry_jobs(status)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_metadata_retry_jobs_next_retry ON metadata_retry_jobs(next_retry_at)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_metadata_retry_jobs_updated ON metadata_retry_jobs(updated_at)"
    )
    repo.conn.commit()


def ensure_book_metadata_schema():
    """
    Remark: book metadata is stored separately from judicial metadata so
    legal books/bare acts are not forced into case-law specific columns.
    """
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS book_metadata (
            document_id TEXT PRIMARY KEY,
            filename TEXT,
            user_id TEXT,
            collection_id TEXT,
            title TEXT,
            language TEXT,
            source_type TEXT,
            page_count INTEGER,
            word_count INTEGER,
            section_count INTEGER,
            article_count INTEGER,
            chapter_count INTEGER,
            part_count INTEGER,
            schedule_count INTEGER,
            toc_count INTEGER,
            inferred_subjects JSONB DEFAULT '[]'::jsonb,
            chapter_titles JSONB DEFAULT '[]'::jsonb,
            toc_entries JSONB DEFAULT '[]'::jsonb,
            act_alias_hits JSONB DEFAULT '[]'::jsonb,
            structure_hints JSONB DEFAULT '{}'::jsonb,
            metadata_confidence TEXT,
            extraction_notes JSONB DEFAULT '{}'::jsonb,
            file_hash TEXT,
            created_at TIMESTAMP,
            ingested_at TIMESTAMP
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_book_metadata_collection ON book_metadata(collection_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_book_metadata_user ON book_metadata(user_id)"
    )
    # Backward-compatible adds for existing deployments.
    cur.execute("ALTER TABLE book_metadata ADD COLUMN IF NOT EXISTS article_count INTEGER")
    cur.execute("ALTER TABLE book_metadata ADD COLUMN IF NOT EXISTS part_count INTEGER")
    cur.execute("ALTER TABLE book_metadata ADD COLUMN IF NOT EXISTS schedule_count INTEGER")
    cur.execute("ALTER TABLE book_metadata ADD COLUMN IF NOT EXISTS toc_count INTEGER")
    cur.execute("ALTER TABLE book_metadata ADD COLUMN IF NOT EXISTS toc_entries JSONB DEFAULT '[]'::jsonb")
    repo.conn.commit()


def ensure_book_section_index_schema():
    """
    Keep an Act+Section lookup index for reference books.
    Remark: this is a normalized retrieval helper (AKN-like footprint) used to
    disambiguate same section numbers across different Acts (e.g., CPC vs CrPC).
    """
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS book_section_index (
            id BIGSERIAL PRIMARY KEY,
            document_id TEXT NOT NULL,
            filename TEXT,
            user_id TEXT,
            collection_id TEXT,
            act_canonical TEXT NOT NULL,
            section_code TEXT NOT NULL,
            parent_section_code TEXT,
            section_type TEXT,
            section_title TEXT,
            chunk_index INTEGER NOT NULL,
            text_preview TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            UNIQUE (document_id, act_canonical, section_code, chunk_index)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_book_section_scope ON book_section_index(collection_id, user_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_book_section_lookup ON book_section_index(act_canonical, section_code)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_book_section_doc ON book_section_index(document_id)"
    )
    cur.execute(
        "ALTER TABLE book_section_index ADD COLUMN IF NOT EXISTS parent_section_code TEXT"
    )
    repo.conn.commit()


def ensure_admin_runtime_settings_schema():
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_runtime_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        """
        INSERT INTO admin_runtime_settings (key, value)
        VALUES ('metadata_retry_paused', 'false')
        ON CONFLICT (key) DO NOTHING
        """
    )
    cur.execute(
        """
        INSERT INTO admin_runtime_settings (key, value)
        VALUES ('metadata_retry_max_queued_per_user', '50')
        ON CONFLICT (key) DO NOTHING
        """
    )
    repo.conn.commit()


def ensure_admin_audit_schema():
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_audit_log (
            id SERIAL PRIMARY KEY,
            actor_user_id TEXT NOT NULL,
            action TEXT NOT NULL,
            target_type TEXT,
            target_id TEXT,
            details JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_admin_audit_created ON admin_audit_log(created_at)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_admin_audit_actor ON admin_audit_log(actor_user_id)"
    )
    repo.conn.commit()


def ensure_podcast_schema():
    """
    Ensure podcast job + review-feedback tables exist.
    Feedback rows capture user corrections/remarks for future prompt quality.
    """
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS podcast_jobs (
            id TEXT PRIMARY KEY,
            notebook_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            speakers INTEGER DEFAULT 2,
            status TEXT NOT NULL DEFAULT 'pending',
            result TEXT,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            audio_path TEXT,
            auto_generate_audio BOOLEAN NOT NULL DEFAULT FALSE
        )
        """
    )
    cur.execute("ALTER TABLE podcast_jobs ADD COLUMN IF NOT EXISTS speakers INTEGER DEFAULT 2")
    cur.execute("ALTER TABLE podcast_jobs ADD COLUMN IF NOT EXISTS result TEXT")
    cur.execute("ALTER TABLE podcast_jobs ADD COLUMN IF NOT EXISTS error TEXT")
    cur.execute("ALTER TABLE podcast_jobs ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    cur.execute("ALTER TABLE podcast_jobs ADD COLUMN IF NOT EXISTS status_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    cur.execute("ALTER TABLE podcast_jobs ADD COLUMN IF NOT EXISTS audio_path TEXT")
    cur.execute(
        "ALTER TABLE podcast_jobs ADD COLUMN IF NOT EXISTS auto_generate_audio BOOLEAN NOT NULL DEFAULT FALSE"
    )
    cur.execute(
        "UPDATE podcast_jobs SET status_updated_at = COALESCE(status_updated_at, created_at, CURRENT_TIMESTAMP)"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_podcast_jobs_user ON podcast_jobs(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_podcast_jobs_notebook ON podcast_jobs(notebook_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_podcast_jobs_status ON podcast_jobs(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_podcast_jobs_created ON podcast_jobs(created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_podcast_jobs_status_updated ON podcast_jobs(status_updated_at)")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS podcast_script_feedback (
            id BIGSERIAL PRIMARY KEY,
            job_id TEXT NOT NULL,
            notebook_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            original_script TEXT,
            edited_script TEXT NOT NULL,
            remarks TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_podcast_feedback_user ON podcast_script_feedback(user_id, created_at DESC)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_podcast_feedback_notebook ON podcast_script_feedback(notebook_id, created_at DESC)"
    )
    repo.conn.commit()


@app.on_event("startup")
def start_ingest_worker():
    ensure_collection_chat_history_schema()
    ensure_chat_archive_schema()
    ensure_chat_jobs_schema()
    ensure_chat_answer_cache_schema()
    ensure_specialization_profiles_schema()
    ensure_metadata_retry_schema()
    ensure_book_metadata_schema()
    ensure_book_section_index_schema()
    ensure_admin_runtime_settings_schema()
    ensure_admin_audit_schema()
    ensure_podcast_schema()
    thread = threading.Thread(
        target=ingest_worker_loop,
        daemon=True,
    )
    thread.start()
    retry_thread = threading.Thread(
        target=metadata_retry_worker_loop,
        daemon=True,
    )
    retry_thread.start()

def run_chat_job(job_id: str, user_id: str):
    repo = get_repo()
    cur = repo.conn.cursor()
    try:
        cur.execute(
            """
            UPDATE chat_jobs
            SET status = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
            """,
            ("running", job_id, user_id),
        )
        repo.conn.commit()

        cur.execute(
            """
            SELECT notebook_id, collection_id, question, include_global, global_sub_collection_ids, specialization
            FROM chat_jobs
            WHERE id = %s AND user_id = %s
            """,
            (job_id, user_id),
        )
        row = cur.fetchone()
        if not row:
            raise Exception("Chat job not found")

        notebook_id, collection_id, question, include_global, global_sub_collection_ids, specialization = row
        normalized_global_filters = _normalize_global_sub_collection_ids(
            cur, list(global_sub_collection_ids or [])
        )
        specialization = (specialization or "").strip() or None

        if collection_id:
            answer_text, answer_payload = _answer_with_cache(
                cur,
                question=question,
                user_id=user_id,
                collection_id=collection_id,
                include_global=bool(include_global),
                global_sub_collection_ids=normalized_global_filters,
                specialization=specialization,
            )
            cur.execute(
                """
                INSERT INTO collection_chat_history (collection_id, user_id, role, content)
                VALUES (%s, %s, %s, %s)
                """,
                (collection_id, user_id, "user", question),
            )
            cur.execute(
                """
                INSERT INTO collection_chat_history (collection_id, user_id, role, content)
                VALUES (%s, %s, %s, %s)
                """,
                (collection_id, user_id, "assistant", answer_text),
            )
        else:
            loaded = load_vectors(notebook_id)
            vectors = loaded[1] if loaded else []
            answer_text, answer_payload = _answer_with_cache(
                cur,
                question=question,
                user_id=user_id,
                notebook_id=notebook_id,
                include_global=bool(include_global),
                global_sub_collection_ids=normalized_global_filters,
                specialization=specialization,
                relevant_chunks=vectors,
            )
            cur.execute(
                """
                INSERT INTO chat_history (notebook_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (notebook_id, "user", question),
            )
            cur.execute(
                """
                INSERT INTO chat_history (notebook_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (notebook_id, "assistant", answer_text),
            )

        cur.execute(
            """
            UPDATE chat_jobs
            SET status = %s, result = %s, result_payload = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
            """,
            ("done", answer_text, Json(answer_payload), job_id, user_id),
        )
        repo.conn.commit()
    except Exception as e:
        repo.conn.rollback()
        cur.execute(
            """
            UPDATE chat_jobs
            SET status = %s, error = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
            """,
            ("error", str(e), job_id, user_id),
        )
        repo.conn.commit()

# ❌ init_db()
# ❌ init_chat_table()
# PostgreSQL schema already exists

# --------------------------------------------------
# Models
# --------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str

class CreateCollectionRequest(BaseModel):
    name: str
    is_global: Optional[bool] = False

class PodcastRequest(BaseModel):
    notebook_id: str
    speakers: int = 2
    auto_generate_audio: bool = True


class PodcastScriptCommitRequest(BaseModel):
    edited_script: str
    remarks: Optional[str] = None
    regenerate_audio: bool = True

class CollectionChatRequest(BaseModel):
    collection_id: str
    question: str
    include_global: bool = False
    global_sub_collection_ids: Optional[List[str]] = None
    specialization: Optional[str] = None

class ChatJobRequest(BaseModel):
    notebook_id: Optional[str] = None
    collection_id: Optional[str] = None
    question: str
    include_global: bool = False
    global_sub_collection_ids: Optional[List[str]] = None
    specialization: Optional[str] = None


class RestoreArchiveRequest(BaseModel):
    archive_session_id: Optional[str] = None
# --------------------------------------------------
# Helpers
# --------------------------------------------------

def verify_password(plain: str, hashed: str) -> bool:
    # bcrypt max = 72 bytes
    return bcrypt.checkpw(
        plain.encode("utf-8")[:72],
        hashed.encode("utf-8"),
    )

def get_current_user_id(x_user_id: str = Header(...)):
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing X-User-Id")
    return x_user_id


def _get_notebook_status(cur, notebook_id: str, user_id: str) -> Optional[str]:
    # Single source of truth for notebook chat readiness checks.
    cur.execute(
        """
        SELECT status
        FROM notebooks
        WHERE notebook_id = %s AND user_id = %s
        """,
        (notebook_id, user_id),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _get_collection_readiness(cur, collection_id: str, user_id: str) -> dict:
    # Aggregated readiness lets collection chat continue on ready docs while others ingest.
    cur.execute(
        """
        SELECT
            COUNT(*)::int AS total_count,
            COUNT(*) FILTER (WHERE COALESCE(n.status, 'ready') = 'ready')::int AS ready_count,
            COUNT(*) FILTER (WHERE COALESCE(n.status, 'ready') IN ('queued', 'processing'))::int AS ingesting_count,
            COUNT(*) FILTER (WHERE COALESCE(n.status, 'ready') = 'failed')::int AS failed_count
        FROM notebooks n
        JOIN collections c ON c.collection_id = n.collection_id
        WHERE n.collection_id = %s
          AND (n.user_id = %s OR c.is_global = TRUE)
        """,
        (collection_id, user_id),
    )
    row = cur.fetchone()
    return {
        "total_count": int(row[0] or 0),
        "ready_count": int(row[1] or 0),
        "ingesting_count": int(row[2] or 0),
        "failed_count": int(row[3] or 0),
    }

# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok"}

# --------------------------------------------------
# Login
# --------------------------------------------------

@app.post("/login")
def login(data: LoginRequest):
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        SELECT id, username, password_hash, role
        FROM users
        WHERE username = %s
        """,
        (data.username,),
    )

    user = cur.fetchone()

    if not user or not verify_password(data.password, user[2]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "user_id": user[0],
        "username": user[1],
        "role": user[3],
    }
# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

def _is_pdf_upload(file: UploadFile) -> bool:
    name = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    return name.endswith(".pdf") and (
        content_type in {"application/pdf", "application/x-pdf", "application/octet-stream", ""}
    )


def _resolve_upload_owner(cur, collection_id: Optional[str], user_id: str):
    owner_id = user_id
    if not collection_id:
        return owner_id

    cur.execute(
        """
        SELECT is_global
        FROM collections
        WHERE collection_id = %s
        """,
        (collection_id,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Collection not found")

    is_global = bool(row[0])
    if not is_global:
        return owner_id

    cur.execute(
        "SELECT role FROM users WHERE id = %s",
        (user_id,),
    )
    role_row = cur.fetchone()
    if not role_row or role_row[0] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only admin can upload to global collection",
        )
    return None


async def _enqueue_pdf_upload(file: UploadFile, collection_id: Optional[str], user_id: str):
    if not _is_pdf_upload(file):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    notebook_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    upload_dir = BASE_DIR / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{notebook_id}.pdf"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    repo = get_repo()
    cur = repo.conn.cursor()

    owner_id = _resolve_upload_owner(cur, collection_id, user_id)

    cur.execute(
        """
        INSERT INTO notebooks
        (notebook_id, filename, user_id, collection_id, status)
        VALUES (%s, %s, %s, %s, 'queued')
        """,
        (notebook_id, file.filename, owner_id, collection_id),
    )

    cur.execute(
        """
        INSERT INTO ingest_jobs
        (job_id, notebook_id, user_id, status)
        VALUES (%s, %s, %s, 'queued')
        """,
        (job_id, notebook_id, user_id),
    )

    repo.conn.commit()
    return {
        "notebook_id": notebook_id,
        "status": "queued",
        "is_global": owner_id is None,
        "filename": file.filename,
    }


@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    collection_id: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user_id),
):
    return await _enqueue_pdf_upload(file, collection_id, user_id)


@app.post("/upload-pdfs")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    collection_id: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user_id),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    uploaded = []
    for file in files:
        uploaded.append(await _enqueue_pdf_upload(file, collection_id, user_id))

    return {
        "status": "queued",
        "count": len(uploaded),
        "uploads": uploaded,
    }

# --------------------------------------------------
# List notebooks
# --------------------------------------------------

@app.get("/notebooks")
def list_notebooks(user_id: str = Depends(get_current_user_id)):
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        SELECT notebook_id, filename, created_at, collection_id, COALESCE(status, 'ready') AS status
        FROM notebooks
        WHERE user_id = %s
        ORDER BY created_at DESC
        """,
        (user_id,),
    )
    rows = cur.fetchall()

    notebooks = []

    for r in rows:
        notebook = {
            "notebook_id": r[0],
            "filename": r[1],
            "created_at": r[2],
            "collection_id": r[3],
            "status": r[4],
        }

        # Get collection name if exists
        if r[3]:
            cur.execute(
                "SELECT name FROM collections WHERE collection_id = %s",
                (r[3],),
            )
            collection = cur.fetchone()
            notebook["collection_name"] = collection[0] if collection else None
        else:
            notebook["collection_name"] = None

        notebooks.append(notebook)

    return notebooks


@app.get("/notebook/{notebook_id}/status")
def get_notebook_status(
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        SELECT notebook_id, filename, COALESCE(status, 'ready') AS status
        FROM notebooks
        WHERE notebook_id = %s AND user_id = %s
        """,
        (notebook_id, user_id),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Notebook not found")

    # Keep status messaging centralized so UI can display consistent gating hints.
    status = row[2]
    can_chat = status == "ready"
    return {
        "notebook_id": row[0],
        "filename": row[1],
        "status": status,
        "can_chat": can_chat,
        "message": (
            "Ingestion complete. You can start chatting."
            if can_chat
            else "Ingestion in progress. Please wait until processing completes."
            if status in ("queued", "processing")
            else "Ingestion failed for this notebook. Re-upload or check worker logs."
        ),
    }
# --------------------------------------------------
# Single PDF Chat
# --------------------------------------------------

@app.post("/pdf-chat")
def pdf_chat(
    payload: dict,
    user_id: str = Depends(get_current_user_id),
):
    notebook_id = payload.get("notebook_id")
    question = payload.get("question")
    include_global = bool(payload.get("include_global", False))
    global_sub_collection_ids = payload.get("global_sub_collection_ids") or []
    specialization = (payload.get("specialization") or "").strip() or None

    if not notebook_id or not question:
        raise HTTPException(status_code=400, detail="Missing data")

    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify notebook ownership
    cur.execute(
        """
        SELECT 1 FROM notebooks
        WHERE notebook_id = %s AND user_id = %s
        """,
        (notebook_id, user_id),
    )
    owned = cur.fetchone()

    if not owned:
        raise HTTPException(status_code=403, detail="Forbidden")

    nb_status = _get_notebook_status(cur, notebook_id, user_id)
    if nb_status in ("queued", "processing"):
        raise HTTPException(
            status_code=409,
            detail="This document is still ingesting. Please wait for completion before chatting.",
        )
    if nb_status == "failed":
        raise HTTPException(
            status_code=409,
            detail="This document ingestion failed. Re-upload or check worker logs.",
        )

    valid_global_sub_collections = _normalize_global_sub_collection_ids(
        cur, global_sub_collection_ids
    )

    loaded = load_vectors(notebook_id)
    if loaded:
        _, metadata = loaded
        vectors = metadata
    else:
        vectors = []

    answer_text, answer_payload = _answer_with_cache(
        cur,
        question=question,
        user_id=user_id,
        notebook_id=notebook_id,
        include_global=include_global,
        global_sub_collection_ids=valid_global_sub_collections,
        specialization=specialization,
        relevant_chunks=vectors,
    )

    # Store chat history
    cur.execute(
        """
        INSERT INTO chat_history (notebook_id, role, content)
        VALUES (%s, %s, %s)
        """,
        (notebook_id, "user", question),
    )
    cur.execute(
        """
        INSERT INTO chat_history (notebook_id, role, content)
        VALUES (%s, %s, %s)
        """,
        (notebook_id, "assistant", answer_text),
    )

    repo.conn.commit()

    return {"answer": answer_text, "result_payload": answer_payload}


# --------------------------------------------------
# Collection Chat
# --------------------------------------------------

@app.post("/collection-chat")
def collection_chat(
    payload: CollectionChatRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Chat with an entire collection of documents
    """
    if not payload.collection_id or not payload.question:
        raise HTTPException(status_code=400, detail="Missing collection_id or question")

    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify collection ownership
    cur.execute(
        """
        SELECT name FROM collections
        WHERE collection_id = %s AND (user_id = %s OR is_global = TRUE)
        """,
        (payload.collection_id, user_id),
    )
    owned = cur.fetchone()

    if not owned:
        raise HTTPException(status_code=403, detail="Collection not found or not owned")

    collection_name = owned[0]
    valid_global_sub_collections = _normalize_global_sub_collection_ids(
        cur, payload.global_sub_collection_ids
    )
    readiness = _get_collection_readiness(cur, payload.collection_id, user_id)
    if readiness["ready_count"] == 0 and readiness["ingesting_count"] > 0:
        raise HTTPException(
            status_code=409,
            detail=(
                "Collection ingestion in progress. No documents are ready yet. "
                "Please wait, then retry."
            ),
        )
    if readiness["ready_count"] == 0 and readiness["total_count"] > 0:
        raise HTTPException(
            status_code=409,
            detail=(
                "No ready documents in this collection. "
                "Some may have failed ingestion; please review and re-upload."
            ),
        )

    # Generate answer using collection-aware pipeline
    answer_text, answer_payload = _answer_with_cache(
        cur,
        question=payload.question,
        user_id=user_id,
        collection_id=payload.collection_id,
        include_global=payload.include_global,
        global_sub_collection_ids=valid_global_sub_collections,
        specialization=(payload.specialization or "").strip() or None,
    )

    # Store in collection chat history
    try:
        cur.execute(
            """
            INSERT INTO collection_chat_history
            (collection_id, user_id, role, content)
            VALUES (%s, %s, %s, %s)
            """,
            (payload.collection_id, user_id, "user", payload.question),
        )
        cur.execute(
            """
            INSERT INTO collection_chat_history
            (collection_id, user_id, role, content)
            VALUES (%s, %s, %s, %s)
            """,
            (payload.collection_id, user_id, "assistant", answer_text),
        )
        repo.conn.commit()
    except Exception as e:
        # Table might not exist (backward compatibility)
        print(f"Note: collection_chat_history table not available: {e}")

    # Get notebook count for info
    notebook_ids = get_collection_notebooks(payload.collection_id, user_id)

    return {
        "answer": answer_text,
        "result_payload": answer_payload,
        "collection_id": payload.collection_id,
        "collection_name": collection_name,
        "sources": len(notebook_ids),
        "ingestion_notice": (
            f"{readiness['ingesting_count']} document(s) still ingesting; "
            f"answer uses {readiness['ready_count']} ready document(s)."
            if readiness["ingesting_count"] > 0 and readiness["ready_count"] > 0
            else None
        ),
    }


@app.post("/chat/submit")
def submit_chat_job(
    payload: ChatJobRequest,
    user_id: str = Depends(get_current_user_id),
):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Missing question")
    if bool(payload.notebook_id) == bool(payload.collection_id):
        raise HTTPException(status_code=400, detail="Provide either notebook_id or collection_id")

    repo = get_repo()
    cur = repo.conn.cursor()

    ingestion_notice = None
    normalized_global_filters = _normalize_global_sub_collection_ids(
        cur, payload.global_sub_collection_ids
    )
    specialization = (payload.specialization or "").strip() or None

    if payload.notebook_id:
        cur.execute(
            """
            SELECT 1 FROM notebooks
            WHERE notebook_id = %s AND user_id = %s
            """,
            (payload.notebook_id, user_id),
        )
        if not cur.fetchone():
            raise HTTPException(status_code=403, detail="Forbidden")

        # Notebook chat is blocked until ingestion completes for that exact file.
        nb_status = _get_notebook_status(cur, payload.notebook_id, user_id)
        if nb_status in ("queued", "processing"):
            raise HTTPException(
                status_code=409,
                detail="This document is still ingesting. Please wait for completion before chatting.",
            )
        if nb_status == "failed":
            raise HTTPException(
                status_code=409,
                detail="This document ingestion failed. Re-upload or check worker logs.",
            )

    if payload.collection_id:
        cur.execute(
            """
            SELECT 1 FROM collections
            WHERE collection_id = %s AND (user_id = %s OR is_global = TRUE)
            """,
            (payload.collection_id, user_id),
        )
        if not cur.fetchone():
            raise HTTPException(status_code=403, detail="Collection not found or not owned")

        # Collection chat is partially available: allowed if at least one doc is ready.
        readiness = _get_collection_readiness(cur, payload.collection_id, user_id)
        if readiness["ready_count"] == 0 and readiness["ingesting_count"] > 0:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Collection ingestion in progress. No documents are ready yet. "
                    "Please wait, then retry."
                ),
            )
        if readiness["ready_count"] == 0 and readiness["total_count"] > 0:
            raise HTTPException(
                status_code=409,
                detail=(
                    "No ready documents in this collection. "
                    "Some may have failed ingestion; please review and re-upload."
                ),
            )
        if readiness["ingesting_count"] > 0 and readiness["ready_count"] > 0:
            ingestion_notice = (
                f"{readiness['ingesting_count']} document(s) are still ingesting. "
                f"Answer is generated from {readiness['ready_count']} ready document(s)."
            )

    job_id = str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO chat_jobs (
            id, user_id, notebook_id, collection_id, question, include_global, global_sub_collection_ids, specialization, status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending')
        """,
        (
            job_id,
            user_id,
            payload.notebook_id,
            payload.collection_id,
            payload.question.strip(),
            bool(payload.include_global),
            Json(normalized_global_filters),
            specialization,
        ),
    )
    repo.conn.commit()

    threading.Thread(
        target=run_chat_job,
        args=(job_id, user_id),
        daemon=True,
    ).start()

    return {"job_id": job_id, "status": "pending", "ingestion_notice": ingestion_notice}


@app.get("/chat/status/{job_id}")
def get_chat_job_status(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        SELECT status, result, error, result_payload
        FROM chat_jobs
        WHERE id = %s AND user_id = %s
        """,
        (job_id, user_id),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    payload = row[3] if len(row) > 3 else None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = None

    return {
        "status": row[0],
        "result": row[1] if row[0] == "done" else None,
        "result_payload": payload if row[0] == "done" else None,
        "error": row[2] if row[0] == "error" else None,
    }
# --------------------------------------------------
# Delete notebook
# --------------------------------------------------

@app.delete("/notebook/{notebook_id}")
def delete_notebook(
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify ownership
    cur.execute(
        """
        SELECT 1 FROM notebooks
        WHERE notebook_id = %s AND user_id = %s
        """,
        (notebook_id, user_id),
    )
    owned = cur.fetchone()

    if not owned:
        raise HTTPException(status_code=404, detail="Not found")

    
    hard_delete_notebook(cur, notebook_id)
    repo.conn.commit()

    return {"status": "deleted", "notebook_id": notebook_id}


# --------------------------------------------------
# Chat history
# --------------------------------------------------

@app.get("/chat-history/{notebook_id}")
def get_chat_history(
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        SELECT role, content, created_at
        FROM chat_history
        WHERE notebook_id = %s
          AND notebook_id IN (
              SELECT notebook_id
              FROM notebooks
              WHERE user_id = %s
          )
        ORDER BY created_at ASC, id ASC
        """,
        (notebook_id, user_id),
    )

    rows = cur.fetchall()

    return [
        {
            "role": r[0],
            "content": r[1],
            "created_at": r[2],
        }
        for r in rows
    ]


@app.get("/chat-history/{notebook_id}/archive")
def get_chat_history_archive(
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM notebooks
        WHERE notebook_id = %s AND user_id = %s
        """,
        (notebook_id, user_id),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=403, detail="Notebook not found or not owned")

    cur.execute(
        """
        SELECT id, role, content, created_at, archived_at, archive_session_id
        FROM chat_history_archive
        WHERE notebook_id = %s AND archived_by = %s
        ORDER BY archived_at DESC, id ASC
        LIMIT 1200
        """,
        (notebook_id, user_id),
    )
    rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "role": r[1],
            "content": r[2],
            "created_at": r[3],
            "archived_at": r[4],
            "archive_session_id": r[5],
        }
        for r in rows
    ]


@app.post("/chat-history/{notebook_id}/archive/restore")
def restore_chat_history_archive(
    notebook_id: str,
    payload: RestoreArchiveRequest,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM notebooks
        WHERE notebook_id = %s AND user_id = %s
        """,
        (notebook_id, user_id),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=403, detail="Notebook not found or not owned")

    session_id = (payload.archive_session_id or "").strip() or None
    if not session_id:
        cur.execute(
            """
            SELECT archive_session_id
            FROM chat_history_archive
            WHERE notebook_id = %s AND archived_by = %s
            ORDER BY archived_at DESC, id DESC
            LIMIT 1
            """,
            (notebook_id, user_id),
        )
        row = cur.fetchone()
        session_id = row[0] if row else None
    if not session_id:
        raise HTTPException(status_code=404, detail="No archived session found")

    cur.execute(
        """
        SELECT role, content, created_at
        FROM chat_history_archive
        WHERE notebook_id = %s
          AND archived_by = %s
          AND archive_session_id = %s
        ORDER BY created_at ASC NULLS LAST, id ASC
        """,
        (notebook_id, user_id, session_id),
    )
    rows = cur.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="Archived session not found")

    cur.execute("DELETE FROM chat_history WHERE notebook_id = %s", (notebook_id,))
    for role, content, created_at in rows:
        cur.execute(
            """
            INSERT INTO chat_history (notebook_id, role, content, created_at)
            VALUES (%s, %s, %s, COALESCE(%s, NOW()))
            """,
            (notebook_id, role, content, created_at),
        )
    repo.conn.commit()
    return {"status": "restored", "scope": "notebook", "notebook_id": notebook_id, "archive_session_id": session_id}


@app.delete("/chat-history/{notebook_id}")
def clear_chat_history(
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM notebooks
        WHERE notebook_id = %s AND user_id = %s
        """,
        (notebook_id, user_id),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=403, detail="Notebook not found or not owned")

    # Move active history to archive, then clear active context.
    archive_session_id = str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO chat_history_archive (notebook_id, role, content, created_at, archived_by, archive_session_id)
        SELECT notebook_id, role, content, created_at, %s, %s
        FROM chat_history
        WHERE notebook_id = %s
        """,
        (user_id, archive_session_id, notebook_id),
    )

    cur.execute(
        "DELETE FROM chat_history WHERE notebook_id = %s",
        (notebook_id,),
    )
    cur.execute(
        """
        DELETE FROM chat_answer_cache
        WHERE scope_type = 'notebook'
          AND scope_id = %s
          AND user_scope = %s
        """,
        (notebook_id, user_id),
    )
    repo.conn.commit()
    return {
        "status": "cleared",
        "scope": "notebook",
        "notebook_id": notebook_id,
        "archived": True,
        "archive_session_id": archive_session_id,
    }


# --------------------------------------------------
# Collection Contents
# --------------------------------------------------

@app.get("/collection/{collection_id}/contents")
def get_collection_contents(
    collection_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Get all notebooks and their info in a collection
    """
    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify ownership
    cur.execute(
        """
        SELECT name, is_global
        FROM collections
        WHERE collection_id = %s AND (user_id = %s OR is_global = TRUE)
        """,
        (collection_id, user_id),
    )
    owned = cur.fetchone()

    if not owned:
        raise HTTPException(status_code=403, detail="Collection not found or not owned")

    collection_name = owned[0]
    is_global_collection = bool(owned[1])

    # Get notebooks in collection
    cur.execute(
        """
        SELECT notebook_id, filename, created_at, COALESCE(status, 'ready') AS status
        FROM notebooks
        WHERE collection_id = %s AND (user_id = %s OR user_id IS NULL)
        ORDER BY created_at
        """,
        (collection_id, user_id),
    )
    notebooks = cur.fetchall()

    notebook_details = []

    for nb in notebooks:
        notebook_id, filename, created_at, status = nb

        meta_path = BASE_DIR / "data" / "faiss" / f"{notebook_id}.json"
        chunk_count = 0

        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                chunk_count = len(metadata)
            except Exception:
                chunk_count = 0

        notebook_details.append(
            {
                "notebook_id": notebook_id,
                "filename": filename,
                "created_at": created_at,
                "chunk_count": chunk_count,
                "status": status,
            }
        )

    # Collection-level readiness summary is used by UI banner + submit gating.
    ready_count = sum(1 for n in notebook_details if n.get("status") == "ready")
    ingesting_count = sum(
        1 for n in notebook_details if n.get("status") in ("queued", "processing")
    )
    failed_count = sum(1 for n in notebook_details if n.get("status") == "failed")

    return {
        "collection_id": collection_id,
        "collection_name": collection_name,
        "is_global": is_global_collection,
        "notebooks": notebook_details,
        "total_notebooks": len(notebook_details),
        "ready_count": ready_count,
        "ingesting_count": ingesting_count,
        "failed_count": failed_count,
    }
# --------------------------------------------------
# Search Across Collection
# --------------------------------------------------

@app.post("/collection/{collection_id}/search")
def search_collection(
    collection_id: str,
    payload: dict,
    user_id: str = Depends(get_current_user_id),
):
    """
    Semantic search across collection (returns raw results, not answers)
    """
    from rag.similarity import similarity_search

    question = payload.get("question", "")
    top_k = payload.get("top_k", 10)

    if not question:
        raise HTTPException(status_code=400, detail="Missing question")

    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify ownership
    cur.execute(
        """
        SELECT name
        FROM collections
        WHERE collection_id = %s AND (user_id = %s OR is_global = TRUE)
        """,
        (collection_id, user_id),
    )
    owned = cur.fetchone()

    if not owned:
        raise HTTPException(status_code=403, detail="Collection not found or not owned")

    collection_name = owned[0]

    # Perform semantic search
    results = similarity_search(
        question=question,
        collection_id=collection_id,
        user_id=user_id,
        TOP_K=top_k,
    )

    # Format results with preview
    formatted_results = []
    for result in results:
        text_preview = result.get("text", "")
        if len(text_preview) > 300:
            text_preview = text_preview[:300] + "..."

        formatted_results.append(
            {
                "text": result.get("text", ""),
                "preview": text_preview,
                "score": round(result.get("score", 0), 4),
                "notebook_id": result.get("notebook_id", ""),
                "source": result.get("source", "Unknown"),
                "chunk_index": result.get("chunk_index", 0),
            }
        )

    return {
        "collection_id": collection_id,
        "collection_name": collection_name,
        "question": question,
        "total_results": len(results),
        "results": formatted_results,
    }


# --------------------------------------------------
# Collection History
# --------------------------------------------------

@app.get("/collection/{collection_id}/history")
def get_collection_chat_history(
    collection_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Get chat history for a collection
    """
    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify ownership
    cur.execute(
        """
        SELECT name
        FROM collections
        WHERE collection_id = %s AND (user_id = %s OR is_global = TRUE)
        """,
        (collection_id, user_id),
    )
    owned = cur.fetchone()

    if not owned:
        raise HTTPException(status_code=403, detail="Collection not found or not owned")

    collection_name = owned[0]

    # Fetch history (table may or may not exist)
    try:
        cur.execute(
            """
            SELECT role, content, created_at
            FROM collection_chat_history
            WHERE collection_id = %s AND user_id = %s
            ORDER BY created_at ASC, id ASC
            """,
            (collection_id, user_id),
        )
        rows = cur.fetchall()

        history = [
            {
                "role": r[0],
                "content": r[1],
                "created_at": r[2],
            }
            for r in rows
        ]
    except Exception:
        history = []

    return {
        "collection_id": collection_id,
        "collection_name": collection_name,
        "history": history,
    }


@app.get("/collection/{collection_id}/history/archive")
def get_collection_chat_history_archive(
    collection_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        SELECT name
        FROM collections
        WHERE collection_id = %s AND (user_id = %s OR is_global = TRUE)
        """,
        (collection_id, user_id),
    )
    owned = cur.fetchone()
    if not owned:
        raise HTTPException(status_code=403, detail="Collection not found or not owned")

    cur.execute(
        """
        SELECT id, role, content, created_at, archived_at, archive_session_id
        FROM collection_chat_history_archive
        WHERE collection_id = %s
          AND user_id = %s
          AND archived_by = %s
        ORDER BY archived_at DESC, id ASC
        LIMIT 2000
        """,
        (collection_id, user_id, user_id),
    )
    rows = cur.fetchall()
    history = [
        {
            "id": r[0],
            "role": r[1],
            "content": r[2],
            "created_at": r[3],
            "archived_at": r[4],
            "archive_session_id": r[5],
        }
        for r in rows
    ]
    return {
        "collection_id": collection_id,
        "collection_name": owned[0],
        "history": history,
    }


@app.post("/collection/{collection_id}/history/archive/restore")
def restore_collection_chat_history_archive(
    collection_id: str,
    payload: RestoreArchiveRequest,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM collections
        WHERE collection_id = %s AND (user_id = %s OR is_global = TRUE)
        """,
        (collection_id, user_id),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=403, detail="Collection not found or not owned")

    session_id = (payload.archive_session_id or "").strip() or None
    if not session_id:
        cur.execute(
            """
            SELECT archive_session_id
            FROM collection_chat_history_archive
            WHERE collection_id = %s
              AND user_id = %s
              AND archived_by = %s
            ORDER BY archived_at DESC, id DESC
            LIMIT 1
            """,
            (collection_id, user_id, user_id),
        )
        row = cur.fetchone()
        session_id = row[0] if row else None
    if not session_id:
        raise HTTPException(status_code=404, detail="No archived session found")

    cur.execute(
        """
        SELECT role, content, created_at
        FROM collection_chat_history_archive
        WHERE collection_id = %s
          AND user_id = %s
          AND archived_by = %s
          AND archive_session_id = %s
        ORDER BY created_at ASC NULLS LAST, id ASC
        """,
        (collection_id, user_id, user_id, session_id),
    )
    rows = cur.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="Archived session not found")

    cur.execute(
        """
        DELETE FROM collection_chat_history
        WHERE collection_id = %s AND user_id = %s
        """,
        (collection_id, user_id),
    )
    for role, content, created_at in rows:
        cur.execute(
            """
            INSERT INTO collection_chat_history (collection_id, user_id, role, content, created_at)
            VALUES (%s, %s, %s, %s, COALESCE(%s, NOW()))
            """,
            (collection_id, user_id, role, content, created_at),
        )
    repo.conn.commit()
    return {"status": "restored", "scope": "collection", "collection_id": collection_id, "archive_session_id": session_id}


@app.delete("/collection/{collection_id}/history")
def clear_collection_chat_history(
    collection_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        SELECT is_global
        FROM collections
        WHERE collection_id = %s AND (user_id = %s OR is_global = TRUE)
        """,
        (collection_id, user_id),
    )
    owned = cur.fetchone()
    if not owned:
        raise HTTPException(status_code=403, detail="Collection not found or not owned")

    is_global_collection = bool(owned[0])

    # Move active history to archive, then clear caller-visible context.
    archive_session_id = str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO collection_chat_history_archive
            (collection_id, user_id, role, content, created_at, archived_by, archive_session_id)
        SELECT collection_id, user_id, role, content, created_at, %s, %s
        FROM collection_chat_history
        WHERE collection_id = %s AND user_id = %s
        """,
        (user_id, archive_session_id, collection_id, user_id),
    )

    cur.execute(
        """
        DELETE FROM collection_chat_history
        WHERE collection_id = %s AND user_id = %s
        """,
        (collection_id, user_id),
    )

    if is_global_collection:
        cur.execute(
            """
            DELETE FROM chat_answer_cache
            WHERE scope_type = 'collection'
              AND scope_id = %s
              AND user_scope IS NULL
            """,
            (collection_id,),
        )
    else:
        cur.execute(
            """
            DELETE FROM chat_answer_cache
            WHERE scope_type = 'collection'
              AND scope_id = %s
              AND user_scope = %s
            """,
            (collection_id, user_id),
        )

    repo.conn.commit()
    return {
        "status": "cleared",
        "scope": "collection",
        "collection_id": collection_id,
        "archived": True,
        "archive_session_id": archive_session_id,
    }


# --------------------------------------------------
# Create Collection
# --------------------------------------------------

@app.post("/collections")
def create_collection(
    data: CreateCollectionRequest,
    user_id: str = Depends(get_current_user_id),
):
    collection_id = str(uuid.uuid4())

    repo = get_repo()
    cur = repo.conn.cursor()

    # Default values
    is_global = False
    category = None

    # If frontend sends is_global and user is admin
    if hasattr(data, "is_global") and data.is_global:
        # You must already have this helper
        require_admin(user_id)
        is_global = True

    cur.execute(
        """
        INSERT INTO collections (collection_id, name, user_id, is_global, category)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            collection_id,
            data.name,
            None if is_global else user_id,
            is_global,
            category,
        ),
    )

    repo.conn.commit()

    return {
        "collection_id": collection_id,
        "name": data.name,
        "is_global": is_global,
    }

# --------------------------------------------------
# List Collections
# --------------------------------------------------

@app.get("/collections")
def list_collections(
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        SELECT 
            c.collection_id,
            c.name,
            c.created_at,
            COUNT(
                CASE
                    WHEN c.is_global = TRUE THEN n.notebook_id
                    WHEN n.user_id = c.user_id THEN n.notebook_id
                    ELSE NULL
                END
            ) AS notebook_count,
            c.is_global
        FROM collections c
        LEFT JOIN notebooks n
          ON c.collection_id = n.collection_id
        WHERE c.user_id = %s OR c.is_global = TRUE
        GROUP BY c.collection_id, c.name, c.created_at, c.is_global
        ORDER BY c.created_at DESC
        """,
        (user_id,),
    )

    rows = cur.fetchall()

    return [
        {
            "collection_id": r[0],
            "name": r[1],
            "created_at": r[2],
            "notebook_count": r[3] or 0,
            "is_global": r[4],
        }
        for r in rows
    ]


# --------------------------------------------------
# Delete Collection
# --------------------------------------------------

@app.delete("/collections/{collection_id}")
def delete_collection(
    collection_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify ownership
    cur.execute(
        """
        SELECT 1
        FROM collections
        WHERE collection_id = %s AND user_id = %s AND is_global = FALSE
        """,
        (collection_id, user_id),
    )
    if not cur.fetchone():
        raise HTTPException(404, "Collection not found")

    # Fetch all notebooks in this collection
    cur.execute(
        """
        SELECT notebook_id
        FROM notebooks
        WHERE collection_id = %s AND user_id = %s
        """,
        (collection_id, user_id),
    )
    notebook_ids = [r[0] for r in cur.fetchall()]

    # HARD DELETE each notebook
    for notebook_id in notebook_ids:
        hard_delete_notebook(cur, notebook_id)

    # Delete collection chat history
    cur.execute(
        "DELETE FROM collection_chat_history WHERE collection_id = %s",
        (collection_id,),
    )

    # Delete collection
    cur.execute(
        "DELETE FROM collections WHERE collection_id = %s AND is_global = FALSE",
        (collection_id,),
    )

    repo.conn.commit()

    return {
        "status": "deleted",
        "collection_id": collection_id,
        "deleted_notebooks": len(notebook_ids),
    }

# --------------------------------------------------
# PODCAST ENDPOINTS
# --------------------------------------------------

@app.post("/podcast/generate")
def generate_podcast(
    data: PodcastRequest,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify notebook ownership
    cur.execute(
        """
        SELECT 1 FROM notebooks
        WHERE notebook_id = %s AND user_id = %s
        """,
        (data.notebook_id, user_id),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=403, detail="Forbidden")

    job_id = str(uuid.uuid4())

    cur.execute(
        """
        INSERT INTO podcast_jobs
        (id, notebook_id, user_id, status, speakers, auto_generate_audio, status_updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        """,
        (
            job_id,
            data.notebook_id,
            user_id,
            "pending",
            data.speakers,
            bool(data.auto_generate_audio),
        ),
    )
    repo.conn.commit()

    threading.Thread(
        target=run_podcast_job_qwen,
        args=(job_id, data.notebook_id, user_id, data.speakers, bool(data.auto_generate_audio)),
        daemon=True,
    ).start()

    return {"job_id": job_id}


@app.get("/podcast/status/{job_id}")
def podcast_status(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        SELECT status, result, error
        FROM podcast_jobs
        WHERE id = %s AND user_id = %s
        """,
        (job_id, user_id),
    )
    row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    status = row[0]
    result = row[1]
    error = row[2]

    if status == "running":
        cur.execute(
            """
            SELECT EXTRACT(EPOCH FROM (NOW() - COALESCE(status_updated_at, created_at)))
            FROM podcast_jobs
            WHERE id = %s AND user_id = %s
            """,
            (job_id, user_id),
        )
        age_row = cur.fetchone()
        age_seconds = float(age_row[0] or 0.0) if age_row else 0.0

        if age_seconds > PODCAST_RUNNING_TIMEOUT_SECONDS:
            timeout_msg = (
                f"Podcast generation timed out after {int(age_seconds)} seconds. "
                "Please retry."
            )
            cur.execute(
                """
                UPDATE podcast_jobs
                SET status = %s, error = %s, status_updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND user_id = %s
                """,
                ("error", timeout_msg, job_id, user_id),
            )
            repo.conn.commit()
            status = "error"
            error = timeout_msg

    return {
        "status": status,
        "result": result if status in ("script_ready", "done", "running") else None,
        "error": error if status == "error" else None,
    }


@app.get("/podcast/latest/{notebook_id}")
def get_latest_podcast(
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        SELECT id, result, speakers
        FROM podcast_jobs
        WHERE notebook_id = %s
          AND user_id = %s
          AND status IN ('script_ready', 'done')
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (notebook_id, user_id),
    )
    row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="No podcast found")

    return {
        "job_id": row[0],
        "result": row[1],
        "speakers": row[2],
    }


@app.get("/podcast/guide")
def podcast_editing_guide():
    return {
        "title": "Podcast Script Review Guide",
        "quick_steps": [
            "Generate script",
            "Review/edit draft",
            "Optionally use spoken formatting commands",
            "Add remarks and save edits",
            "Generate final audio from approved script",
        ],
        "commands": {
            "punctuation": [
                "full stop -> .",
                "comma -> ,",
                "colon -> :",
                "semi colon -> ;",
                "question mark -> ?",
            ],
            "structure": [
                "open bracket / in bracket -> (",
                "close bracket -> )",
                "next para / next paragraph -> new paragraph",
                "next line -> line break",
            ],
            "table": [
                "start table",
                "next column",
                "next row",
                "end table",
            ],
        },
        "example": (
            "Rahul: Start table Section next column Punishment next row 420 IPC "
            "next column Up to 7 years end table full stop"
        ),
    }


@app.post("/podcast/commit/{job_id}")
def commit_podcast_script(
    job_id: str,
    data: PodcastScriptCommitRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Save reviewed script edits, persist remarks, and optionally regenerate audio.
    """
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        SELECT notebook_id, speakers, result
        FROM podcast_jobs
        WHERE id = %s AND user_id = %s
        """,
        (job_id, user_id),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Podcast job not found")

    notebook_id, speakers, original_script = row[0], int(row[1] or 2), row[2] or ""

    try:
        normalized_script, validation = normalize_and_validate_podcast_script(
            script=data.edited_script,
            speakers=speakers,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    cur.execute(
        """
        INSERT INTO podcast_script_feedback
        (job_id, notebook_id, user_id, original_script, edited_script, remarks)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            job_id,
            notebook_id,
            user_id,
            original_script,
            normalized_script,
            (data.remarks or "").strip() or None,
        ),
    )
    cur.execute(
        """
        UPDATE podcast_jobs
        SET result = %s, error = NULL
        WHERE id = %s AND user_id = %s
        """,
        (normalized_script, job_id, user_id),
    )

    if data.regenerate_audio:
        cur.execute(
            """
            UPDATE podcast_jobs
            SET status = %s, audio_path = NULL, error = NULL, status_updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
            """,
            ("running", job_id, user_id),
        )
    else:
        cur.execute(
            """
            UPDATE podcast_jobs
            SET status = %s, status_updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
            """,
            ("script_ready", job_id, user_id),
        )
    repo.conn.commit()

    if data.regenerate_audio:
        threading.Thread(
            target=synthesize_podcast_audio,
            args=(job_id, user_id, normalized_script, speakers),
            daemon=True,
        ).start()

    return {
        "job_id": job_id,
        "status": "running" if data.regenerate_audio else "script_ready",
        "result": normalized_script,
        "remarks_saved": bool((data.remarks or "").strip()),
        "validation": validation,
    }


@app.get("/podcast/audio/{job_id}")
def get_podcast_audio(job_id: str):
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        "SELECT audio_path FROM podcast_jobs WHERE id = %s",
        (job_id,),
    )
    row = cur.fetchone()

    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Audio not ready")

    audio_path = Path(row[0])
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing")

    file_size = os.path.getsize(audio_path)

    def iterfile():
        with open(audio_path, "rb") as f:
            yield from f

    return StreamingResponse(
        iterfile(),
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Disposition": f'inline; filename="{audio_path.name}"',
        },
    )


# --------------------------------------------------
# REMOVE NOTEBOOK FROM COLLECTION
# --------------------------------------------------

@app.delete("/collection/{collection_id}/notebook/{notebook_id}")
def remove_notebook_from_collection(
    collection_id: str,
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()

    # Remark: allow two valid managers:
    # - owner of a non-global collection
    # - admin for global collections
    cur.execute(
        """
        SELECT is_global, user_id
        FROM collections
        WHERE collection_id = %s
        """,
        (collection_id,),
    )
    collection_row = cur.fetchone()
    if not collection_row:
        raise HTTPException(403, "Collection not found or not owned")

    is_global_collection = bool(collection_row[0])
    collection_owner = collection_row[1]

    cur.execute("SELECT role FROM users WHERE id = %s", (user_id,))
    role_row = cur.fetchone()
    requester_is_admin = bool(role_row and role_row[0] == "admin")

    if is_global_collection:
        if not requester_is_admin:
            raise HTTPException(403, "Collection not found or not owned")
    else:
        if collection_owner != user_id:
            raise HTTPException(403, "Collection not found or not owned")

    # Verify notebook belongs to this collection
    if is_global_collection:
        # Remark: global collection documents can be ownerless (user_id NULL).
        cur.execute(
            """
            SELECT 1
            FROM notebooks
            WHERE notebook_id = %s
              AND collection_id = %s
            """,
            (notebook_id, collection_id),
        )
    else:
        cur.execute(
            """
            SELECT 1
            FROM notebooks
            WHERE notebook_id = %s
              AND collection_id = %s
              AND user_id = %s
            """,
            (notebook_id, collection_id, user_id),
        )
    if not cur.fetchone():
        raise HTTPException(404, "Notebook not found in collection")

    # HARD DELETE
    hard_delete_notebook(cur, notebook_id)

    repo.conn.commit()

    return {
        "status": "deleted",
        "notebook_id": notebook_id,
        "collection_id": collection_id,
    }
# --------------------------------------------------
# ADMIN ONLY CREATE USER ENDPOINT
# --------------------------------------------------
class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class RetryQueueCapRequest(BaseModel):
    max_queued_per_user: int


class PurgeFailedRetriesRequest(BaseModel):
    older_than_days: int = 1

class RequeueStuckIngestRequest(BaseModel):
    older_than_minutes: int = 20


class NormalizeBookTitlesRequest(BaseModel):
    dry_run: bool = False


class AdminBookTitleUpdateRequest(BaseModel):
    title: str

def require_admin(user_id: str):
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute("SELECT role FROM users WHERE id = %s", (user_id,))
    row = cur.fetchone()

    if not row or row[0] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")


def _get_runtime_setting(cur, key: str, default: str) -> str:
    cur.execute("SELECT value FROM admin_runtime_settings WHERE key = %s", (key,))
    row = cur.fetchone()
    return row[0] if row else default


def _set_runtime_setting(cur, key: str, value: str):
    cur.execute(
        """
        INSERT INTO admin_runtime_settings (key, value, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (key) DO UPDATE
        SET value = EXCLUDED.value,
            updated_at = NOW()
        """,
        (key, value),
    )


def _admin_audit(cur, actor_user_id: str, action: str, target_type: str | None, target_id: str | None, details: dict | None = None):
    details = details or {}
    cur.execute(
        """
        INSERT INTO admin_audit_log (actor_user_id, action, target_type, target_id, details)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        """,
        (actor_user_id, action, target_type, target_id, json.dumps(details)),
    )


def _titlecase_legal_name(text: str) -> str:
    acronyms = {"IPC", "CRPC", "CPC", "BNSS", "BNS", "BSA", "NI", "NIA", "GST", "IT", "MVA"}
    small = {"of", "and", "the", "to", "in", "for", "on", "with", "by", "at"}
    parts = re.split(r"(\s+)", text.strip())
    out = []
    word_idx = 0
    for p in parts:
        if not p or p.isspace():
            out.append(p)
            continue
        core = p.strip()
        up = core.upper()
        if up in acronyms:
            out.append(up)
        elif word_idx > 0 and core.lower() in small:
            out.append(core.lower())
        else:
            out.append(core[:1].upper() + core[1:].lower())
        word_idx += 1
    return "".join(out).strip()


def _suggest_book_title(filename: str | None, act_names_json) -> str:
    if isinstance(act_names_json, list) and act_names_json:
        first = act_names_json[0]
        if isinstance(first, dict):
            act_name = (first.get("act") or "").strip()
            if act_name:
                return act_name
        if isinstance(first, str) and first.strip():
            return first.strip()

    base = (filename or "Untitled Book").strip()
    base = re.sub(r"\.pdf$", "", base, flags=re.IGNORECASE)
    base = base.replace("_", " ")
    base = re.sub(r"\s*,\s*", ", ", base)
    base = re.sub(r"\s+", " ", base).strip(" -_")
    if not base:
        base = "Untitled Book"
    return _titlecase_legal_name(base)


@app.get("/admin/global-book-naming-guide")
def admin_global_book_naming_guide(user_id: str = Depends(get_current_user_id)):
    require_admin(user_id)
    return {
        "title_format": "<Canonical Act/Book Name>, <Year>",
        "rules": [
            "Use canonical legal name when identifiable from act catalog.",
            "Preserve legal acronyms in uppercase (IPC, CrPC, CPC, BNSS, BNS, BSA, NI).",
            "Use spaces instead of underscores.",
            "Keep year as 4 digits if available.",
            "Avoid source labels, URLs, download markers, OCR noise.",
            "One title per book across global collection for deterministic retrieval.",
        ],
        "examples": [
            {"input": "negotiable_instruments_act,_1881.pdf", "output": "Negotiable Instruments Act, 1881"},
            {"input": "the_code_of_criminal_procedure,_1973.pdf", "output": "Code of Criminal Procedure, 1973"},
            {"input": "constitution of india.PDF", "output": "Constitution of India"},
        ],
    }


@app.get("/admin/global-collection/{collection_id}/books")
def admin_list_global_collection_books(
    collection_id: str,
    user_id: str = Depends(get_current_user_id),
):
    require_admin(user_id)
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        SELECT name
        FROM collections
        WHERE collection_id = %s AND is_global = TRUE
        """,
        (collection_id,),
    )
    c = cur.fetchone()
    if not c:
        raise HTTPException(status_code=404, detail="Global collection not found")

    cur.execute(
        """
        SELECT
            n.notebook_id,
            n.filename,
            COALESCE(bm.title, '') AS current_title,
            dm.act_names
        FROM notebooks n
        LEFT JOIN book_metadata bm ON bm.document_id = n.notebook_id
        LEFT JOIN document_metadata dm ON dm.document_id = n.notebook_id
        WHERE n.collection_id = %s
        ORDER BY n.created_at DESC
        """,
        (collection_id,),
    )
    rows = cur.fetchall()
    books = []
    for notebook_id, filename, current_title, act_names in rows:
        suggested = _suggest_book_title(filename, act_names)
        books.append(
            {
                "document_id": notebook_id,
                "filename": filename,
                "current_title": current_title or None,
                "suggested_title": suggested,
                "needs_update": (current_title or "").strip() != suggested.strip(),
            }
        )

    return {"collection_id": collection_id, "collection_name": c[0], "books": books}


@app.post("/admin/global-collection/{collection_id}/books/normalize-titles")
def admin_normalize_global_book_titles(
    collection_id: str,
    payload: NormalizeBookTitlesRequest,
    user_id: str = Depends(get_current_user_id),
):
    require_admin(user_id)
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM collections
        WHERE collection_id = %s AND is_global = TRUE
        """,
        (collection_id,),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail="Global collection not found")

    cur.execute(
        """
        SELECT
            n.notebook_id,
            n.filename,
            COALESCE(bm.title, '') AS current_title,
            dm.act_names,
            n.user_id
        FROM notebooks n
        LEFT JOIN book_metadata bm ON bm.document_id = n.notebook_id
        LEFT JOIN document_metadata dm ON dm.document_id = n.notebook_id
        WHERE n.collection_id = %s
        """,
        (collection_id,),
    )
    rows = cur.fetchall()
    changes = []
    for document_id, filename, current_title, act_names, owner_id in rows:
        suggested = _suggest_book_title(filename, act_names)
        if (current_title or "").strip() == suggested.strip():
            continue
        changes.append(
            {
                "document_id": document_id,
                "filename": filename,
                "old_title": current_title or None,
                "new_title": suggested,
            }
        )
        if payload.dry_run:
            continue
        cur.execute(
            """
            INSERT INTO book_metadata (document_id, filename, user_id, collection_id, title, ingested_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (document_id) DO UPDATE
            SET title = EXCLUDED.title,
                filename = COALESCE(book_metadata.filename, EXCLUDED.filename),
                user_id = COALESCE(book_metadata.user_id, EXCLUDED.user_id),
                collection_id = COALESCE(book_metadata.collection_id, EXCLUDED.collection_id),
                ingested_at = NOW()
            """,
            (document_id, filename, owner_id, collection_id, suggested),
        )

    _admin_audit(
        cur,
        actor_user_id=user_id,
        action="normalize_global_book_titles",
        target_type="collection",
        target_id=collection_id,
        details={"dry_run": payload.dry_run, "updated_count": len(changes)},
    )
    if not payload.dry_run:
        repo.conn.commit()
    else:
        repo.conn.rollback()
    return {"collection_id": collection_id, "dry_run": payload.dry_run, "updated_count": len(changes), "changes": changes[:200]}


@app.put("/admin/global-book/{document_id}/title")
def admin_update_global_book_title(
    document_id: str,
    payload: AdminBookTitleUpdateRequest,
    user_id: str = Depends(get_current_user_id),
):
    require_admin(user_id)
    new_title = (payload.title or "").strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        SELECT n.collection_id, n.filename, n.user_id
        FROM notebooks n
        JOIN collections c ON c.collection_id = n.collection_id
        WHERE n.notebook_id = %s
          AND c.is_global = TRUE
        """,
        (document_id,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Global book not found")
    collection_id, filename, owner_id = row

    cur.execute(
        """
        INSERT INTO book_metadata (document_id, filename, user_id, collection_id, title, ingested_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (document_id) DO UPDATE
        SET title = EXCLUDED.title,
            filename = COALESCE(book_metadata.filename, EXCLUDED.filename),
            user_id = COALESCE(book_metadata.user_id, EXCLUDED.user_id),
            collection_id = COALESCE(book_metadata.collection_id, EXCLUDED.collection_id),
            ingested_at = NOW()
        """,
        (document_id, filename, owner_id, collection_id, new_title),
    )
    _admin_audit(
        cur,
        actor_user_id=user_id,
        action="update_global_book_title",
        target_type="book",
        target_id=document_id,
        details={"title": new_title},
    )
    repo.conn.commit()
    return {"status": "updated", "document_id": document_id, "title": new_title}

@app.post("/admin/create-user")
def create_user(
    data: CreateUserRequest,
    user_id: str = Depends(get_current_user_id),
):
    require_admin(user_id)

    repo = get_repo()
    cur = repo.conn.cursor()

    hashed = bcrypt.hashpw(
        data.password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    new_id = str(uuid.uuid4())

    cur.execute("""
        INSERT INTO users (id, username, password_hash, role)
        VALUES (%s, %s, %s, %s)
    """, (new_id, data.username, hashed, data.role))
    _admin_audit(
        cur,
        actor_user_id=user_id,
        action="create_user",
        target_type="user",
        target_id=new_id,
        details={"username": data.username, "role": data.role},
    )

    repo.conn.commit()

    return {
        "user_id": new_id,
        "username": data.username,
        "role": data.role
    }


@app.get("/admin/users")
def list_users_admin(user_id: str = Depends(get_current_user_id)):
    require_admin(user_id)
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        SELECT
            u.id,
            u.username,
            u.role,
            COALESCE(n.nb_count, 0) AS notebook_count,
            COALESCE(c.col_count, 0) AS collection_count
        FROM users u
        LEFT JOIN (
            SELECT user_id, COUNT(*) AS nb_count
            FROM notebooks
            GROUP BY user_id
        ) n ON n.user_id = u.id
        LEFT JOIN (
            SELECT user_id, COUNT(*) AS col_count
            FROM collections
            WHERE is_global = FALSE
            GROUP BY user_id
        ) c ON c.user_id = u.id
        ORDER BY u.username
        """
    )
    rows = cur.fetchall()
    return [
        {
            "user_id": r[0],
            "username": r[1],
            "role": r[2],
            "notebook_count": int(r[3] or 0),
            "collection_count": int(r[4] or 0),
        }
        for r in rows
    ]


@app.delete("/admin/users/{target_user_id}")
def delete_user_admin(
    target_user_id: str,
    user_id: str = Depends(get_current_user_id),
):
    require_admin(user_id)
    if target_user_id == user_id:
        raise HTTPException(status_code=400, detail="Admin cannot delete self")

    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute("SELECT role FROM users WHERE id = %s", (target_user_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    if row[0] == "admin":
        raise HTTPException(status_code=400, detail="Cannot delete another admin")

    cur.execute("SELECT notebook_id FROM notebooks WHERE user_id = %s", (target_user_id,))
    notebook_ids = [r[0] for r in cur.fetchall()]
    for notebook_id in notebook_ids:
        hard_delete_notebook(cur, notebook_id)

    cur.execute("DELETE FROM metadata_retry_jobs WHERE user_id = %s", (target_user_id,))
    cur.execute("DELETE FROM chat_jobs WHERE user_id = %s", (target_user_id,))
    cur.execute("DELETE FROM podcast_jobs WHERE user_id = %s", (target_user_id,))
    cur.execute("DELETE FROM collection_chat_history WHERE user_id = %s", (target_user_id,))
    cur.execute("DELETE FROM collections WHERE user_id = %s", (target_user_id,))
    cur.execute("DELETE FROM users WHERE id = %s", (target_user_id,))
    _admin_audit(
        cur,
        actor_user_id=user_id,
        action="delete_user",
        target_type="user",
        target_id=target_user_id,
        details={"deleted_notebooks": len(notebook_ids)},
    )
    repo.conn.commit()

    return {
        "status": "deleted",
        "user_id": target_user_id,
        "deleted_notebooks": len(notebook_ids),
    }


@app.get("/admin/workers/overview")
def admin_workers_overview(user_id: str = Depends(get_current_user_id)):
    require_admin(user_id)
    repo = get_repo()
    cur = repo.conn.cursor()

    paused = _get_runtime_setting(cur, "metadata_retry_paused", "false").lower() == "true"
    max_queued = int(_get_runtime_setting(cur, "metadata_retry_max_queued_per_user", "50"))

    cur.execute(
        """
        SELECT status, COUNT(*)
        FROM ingest_jobs
        GROUP BY status
        """
    )
    ingest_counts = {r[0]: int(r[1]) for r in cur.fetchall()}
    # Remark: split active vs stale processing to avoid confusion in admin status view.
    cur.execute(
        """
        SELECT
            SUM(CASE WHEN status = 'processing'
                      AND COALESCE(started_at, created_at) >= NOW() - INTERVAL '20 minutes'
                     THEN 1 ELSE 0 END) AS active_processing,
            SUM(CASE WHEN status = 'processing'
                      AND COALESCE(started_at, created_at) < NOW() - INTERVAL '20 minutes'
                     THEN 1 ELSE 0 END) AS stale_processing
        FROM ingest_jobs
        """
    )
    proc_split = cur.fetchone() or (0, 0)
    ingest_counts["active_processing"] = int(proc_split[0] or 0)
    ingest_counts["stale_processing"] = int(proc_split[1] or 0)

    cur.execute(
        """
        SELECT status, COUNT(*)
        FROM metadata_retry_jobs
        GROUP BY status
        """
    )
    retry_counts = {r[0]: int(r[1]) for r in cur.fetchall()}

    cur.execute(
        """
        SELECT status, COUNT(*)
        FROM chat_jobs
        GROUP BY status
        """
    )
    chat_counts = {r[0]: int(r[1]) for r in cur.fetchall()}

    cur.execute(
        """
        SELECT status, COUNT(*)
        FROM podcast_jobs
        GROUP BY status
        """
    )
    podcast_counts = {r[0]: int(r[1]) for r in cur.fetchall()}

    cur.execute(
        """
        SELECT document_id, attempts, status, last_error, updated_at
        FROM metadata_retry_jobs
        ORDER BY updated_at DESC
        LIMIT 20
        """
    )
    recent_retry = [
        {
            "document_id": r[0],
            "attempts": int(r[1] or 0),
            "status": r[2],
            "last_error": r[3],
            "updated_at": str(r[4]),
        }
        for r in cur.fetchall()
    ]

    cur.execute(
        """
        SELECT actor_user_id, action, target_type, target_id, details, created_at
        FROM admin_audit_log
        ORDER BY created_at DESC
        LIMIT 30
        """
    )
    audit = [
        {
            "actor_user_id": r[0],
            "action": r[1],
            "target_type": r[2],
            "target_id": r[3],
            "details": r[4] if isinstance(r[4], dict) else {},
            "created_at": str(r[5]),
        }
        for r in cur.fetchall()
    ]

    return {
        "settings": {
            "metadata_retry_paused": paused,
            "metadata_retry_max_queued_per_user": max_queued,
        },
        "workers": {
            "ingest_worker": {"enabled": True},
            "metadata_retry_worker": {"enabled": not paused},
        },
        "queues": {
            "ingest_jobs": ingest_counts,
            "metadata_retry_jobs": retry_counts,
            "chat_jobs": chat_counts,
            "podcast_jobs": podcast_counts,
        },
        "recent_metadata_retries": recent_retry,
        "recent_admin_audit": audit,
    }


@app.post("/admin/workers/metadata-retry/pause")
def pause_metadata_retry_worker(user_id: str = Depends(get_current_user_id)):
    require_admin(user_id)
    repo = get_repo()
    cur = repo.conn.cursor()
    _set_runtime_setting(cur, "metadata_retry_paused", "true")
    _admin_audit(cur, user_id, "pause_metadata_retry_worker", "worker", "metadata_retry", {})
    repo.conn.commit()
    return {"status": "paused"}


@app.post("/admin/workers/metadata-retry/resume")
def resume_metadata_retry_worker(user_id: str = Depends(get_current_user_id)):
    require_admin(user_id)
    repo = get_repo()
    cur = repo.conn.cursor()
    _set_runtime_setting(cur, "metadata_retry_paused", "false")
    _admin_audit(cur, user_id, "resume_metadata_retry_worker", "worker", "metadata_retry", {})
    repo.conn.commit()
    return {"status": "running"}


@app.post("/admin/workers/metadata-retry/purge-failed")
def purge_failed_metadata_retries(
    payload: PurgeFailedRetriesRequest,
    user_id: str = Depends(get_current_user_id),
):
    require_admin(user_id)
    older_days = max(0, int(payload.older_than_days))
    repo = get_repo()
    cur = repo.conn.cursor()
    cur.execute(
        """
        DELETE FROM metadata_retry_jobs
        WHERE status = 'failed'
          AND updated_at < NOW() - make_interval(days => %s)
        """,
        (older_days,),
    )
    deleted = cur.rowcount
    _admin_audit(
        cur,
        user_id,
        "purge_failed_metadata_retries",
        "queue",
        "metadata_retry_jobs",
        {"older_than_days": older_days, "deleted_jobs": int(deleted)},
    )
    repo.conn.commit()
    return {"status": "deleted", "deleted_jobs": int(deleted)}


@app.post("/admin/workers/metadata-retry/set-cap")
def set_metadata_retry_cap(
    payload: RetryQueueCapRequest,
    user_id: str = Depends(get_current_user_id),
):
    require_admin(user_id)
    cap = max(1, min(500, int(payload.max_queued_per_user)))
    repo = get_repo()
    cur = repo.conn.cursor()
    _set_runtime_setting(cur, "metadata_retry_max_queued_per_user", str(cap))
    _admin_audit(
        cur,
        user_id,
        "set_metadata_retry_cap",
        "setting",
        "metadata_retry_max_queued_per_user",
        {"value": cap},
    )
    repo.conn.commit()
    return {"status": "updated", "metadata_retry_max_queued_per_user": cap}


@app.post("/admin/workers/ingest/requeue-stuck")
def requeue_stuck_ingest_jobs(
    payload: RequeueStuckIngestRequest,
    user_id: str = Depends(get_current_user_id),
):
    require_admin(user_id)
    older_than_minutes = max(0, min(24 * 60, int(payload.older_than_minutes)))
    repo = get_repo()
    cur = repo.conn.cursor()

    # Remark: manual admin recovery to unblock jobs that stayed in processing.
    cur.execute(
        """
        WITH candidates AS (
            SELECT job_id, notebook_id
            FROM ingest_jobs
            WHERE status = 'processing'
              AND finished_at IS NULL
              AND (
                  started_at IS NULL
                  OR started_at < NOW() - make_interval(mins => %s)
              )
            FOR UPDATE SKIP LOCKED
        ),
        requeued_jobs AS (
            UPDATE ingest_jobs j
            SET status = 'queued',
                started_at = NULL,
                error = 'Manual admin requeue from processing'
            FROM candidates c
            WHERE j.job_id = c.job_id
            RETURNING j.job_id, j.notebook_id
        ),
        requeued_notebooks AS (
            UPDATE notebooks n
            SET status = 'queued'
            FROM requeued_jobs rj
            WHERE n.notebook_id = rj.notebook_id
              AND n.status = 'processing'
            RETURNING n.notebook_id
        )
        SELECT
            (SELECT COUNT(*) FROM requeued_jobs) AS job_count,
            (SELECT COUNT(*) FROM requeued_notebooks) AS notebook_count
        """
        ,
        (older_than_minutes,),
    )
    counts = cur.fetchone() or (0, 0)
    requeued_jobs = int(counts[0] or 0)
    requeued_notebooks = int(counts[1] or 0)

    _admin_audit(
        cur,
        user_id,
        "requeue_stuck_ingest_jobs",
        "queue",
        "ingest_jobs",
        {
            "older_than_minutes": older_than_minutes,
            "requeued_jobs": requeued_jobs,
            "requeued_notebooks": requeued_notebooks,
        },
    )
    repo.conn.commit()
    return {
        "status": "ok",
        "older_than_minutes": older_than_minutes,
        "requeued_jobs": requeued_jobs,
        "requeued_notebooks": requeued_notebooks,
    }

# --------------------------------------------------
# REBUILD GLOBAL INDEX ENDPOINT
# --------------------------------------------------

@app.post("/admin/rebuild-global-index")
def rebuild_global_index(user_id: str = Depends(get_current_user_id)):
    require_admin(user_id)

    from rag.vector_store import build_global_index
    build_global_index()
    repo = get_repo()
    cur = repo.conn.cursor()
    _admin_audit(cur, user_id, "rebuild_global_index", "worker", "global_index", {})
    repo.conn.commit()

    return {"status": "Global index rebuilt"}
# --------------------------------------------------
# delete GLOBAL collection from admin ENDPOINT
# --------------------------------------------------

@app.delete("/admin/global-collection/{collection_id}")
def delete_global_collection(
    collection_id: str,
    user_id: str = Depends(get_current_user_id),
):
    require_admin(user_id)

    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify it is actually global
    cur.execute(
        """
        SELECT 1
        FROM collections
        WHERE collection_id = %s AND is_global = TRUE
        """,
        (collection_id,),
    )

    if not cur.fetchone():
        raise HTTPException(404, "Global collection not found")

    # Get all notebooks in this collection
    cur.execute(
        """
        SELECT notebook_id
        FROM notebooks
        WHERE collection_id = %s
        """,
        (collection_id,),
    )

    notebook_ids = [r[0] for r in cur.fetchall()]

    # Hard delete each notebook
    for notebook_id in notebook_ids:
        hard_delete_notebook(cur, notebook_id)

    # Delete chat history
    cur.execute(
        "DELETE FROM collection_chat_history WHERE collection_id = %s",
        (collection_id,),
    )

    # Delete collection
    cur.execute(
        "DELETE FROM collections WHERE collection_id = %s",
        (collection_id,),
    )
    _admin_audit(
        cur,
        actor_user_id=user_id,
        action="delete_global_collection",
        target_type="collection",
        target_id=collection_id,
        details={"deleted_notebooks": len(notebook_ids)},
    )

    repo.conn.commit()

    return {
        "status": "deleted",
        "collection_id": collection_id,
        "deleted_notebooks": len(notebook_ids),
    }
