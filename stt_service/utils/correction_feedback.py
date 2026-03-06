"""
User correction feedback persistence for STT post-edit rectification.
"""

from __future__ import annotations

import csv
import difflib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_FEEDBACK_PATH = os.getenv(
    "STT_FEEDBACK_CSV",
    "/home/ubuntu/openbooklm-local/data/stt-transcripts/stt_user_feedback.csv",
)
DEFAULT_FEEDBACK_TABLE = os.getenv("STT_FEEDBACK_TABLE", "stt_user_feedback")


def _feedback_db_dsn() -> str:
    dsn = (os.getenv("STT_FEEDBACK_DB_DSN") or os.getenv("DATABASE_URL") or "").strip()
    # Ignore non-Postgres URLs.
    if not dsn.lower().startswith(("postgres://", "postgresql://")):
        return ""
    return dsn


def _ensure_feedback_table(conn, table_name: str) -> str:
    safe_table = "".join(ch for ch in (table_name or "") if ch.isalnum() or ch == "_").strip("_")
    if not safe_table:
        safe_table = "stt_user_feedback"
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {safe_table} (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                original_transcript TEXT NOT NULL,
                edited_transcript TEXT NOT NULL,
                remarks TEXT,
                replacements_json JSONB NOT NULL DEFAULT '[]'::jsonb
            );
            """
        )
    conn.commit()
    return safe_table


def _save_feedback_postgres(
    original_transcript: str,
    edited_transcript: str,
    remarks: Optional[str],
    pairs: List[Dict[str, str]],
    *,
    dsn: str,
    table_name: str,
) -> Optional[Dict]:
    try:
        import psycopg2
        from psycopg2.extras import Json
    except Exception:
        return None

    try:
        with psycopg2.connect(dsn) as conn:
            safe_table = _ensure_feedback_table(conn, table_name)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {safe_table}
                        (original_transcript, edited_transcript, remarks, replacements_json)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        (original_transcript or "").strip(),
                        (edited_transcript or "").strip(),
                        (remarks or "").strip(),
                        Json(pairs),
                    ),
                )
            conn.commit()
        return {"feedback_db_table": safe_table, "replacement_pairs": pairs}
    except Exception:
        return None


def _load_feedback_phrase_map_postgres(*, dsn: str, table_name: str, max_rows: int) -> Optional[Dict[str, str]]:
    try:
        import psycopg2
    except Exception:
        return None

    phrase_map: Dict[str, str] = {}
    try:
        with psycopg2.connect(dsn) as conn:
            safe_table = _ensure_feedback_table(conn, table_name)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT replacements_json
                    FROM {safe_table}
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (max_rows,),
                )
                rows = cur.fetchall()

        # Apply older->newer so latest correction wins.
        for (raw_pairs,) in reversed(rows):
            pairs = raw_pairs if isinstance(raw_pairs, list) else []
            for pair in pairs:
                wrong = str((pair or {}).get("wrong") or "").strip()
                correct = str((pair or {}).get("correct") or "").strip()
                if len(wrong) < 3 or len(correct) < 1:
                    continue
                phrase_map[wrong.lower()] = correct
        return phrase_map
    except Exception:
        return None


def _ensure_feedback_file(path: str) -> Path:
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    if not fp.exists():
        with open(fp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "original_transcript",
                    "edited_transcript",
                    "remarks",
                    "replacements_json",
                ],
            )
            writer.writeheader()
    return fp


def extract_replacement_pairs(original: str, edited: str, max_pairs: int = 12) -> List[Dict[str, str]]:
    """
    Build phrase-level replacement candidates from user edits.
    Conservative extraction to reduce accidental over-corrections.
    """
    old_tokens = (original or "").split()
    new_tokens = (edited or "").split()
    seq = difflib.SequenceMatcher(a=old_tokens, b=new_tokens)
    pairs: List[Dict[str, str]] = []

    for op, i1, i2, j1, j2 in seq.get_opcodes():
        if op != "replace":
            continue
        old_phrase = " ".join(old_tokens[i1:i2]).strip()
        new_phrase = " ".join(new_tokens[j1:j2]).strip()
        if not old_phrase or not new_phrase:
            continue
        if len(old_phrase) > 80 or len(new_phrase) > 80:
            continue
        if old_phrase.lower() == new_phrase.lower():
            continue
        pairs.append({"wrong": old_phrase, "correct": new_phrase})
        if len(pairs) >= max_pairs:
            break
    return pairs


def save_feedback(
    original_transcript: str,
    edited_transcript: str,
    remarks: str | None = None,
    *,
    path: str = DEFAULT_FEEDBACK_PATH,
) -> Dict:
    pairs = extract_replacement_pairs(original_transcript, edited_transcript)
    dsn = _feedback_db_dsn()
    if dsn:
        db_saved = _save_feedback_postgres(
            original_transcript=original_transcript,
            edited_transcript=edited_transcript,
            remarks=remarks,
            pairs=pairs,
            dsn=dsn,
            table_name=DEFAULT_FEEDBACK_TABLE,
        )
        if db_saved is not None:
            return db_saved

    fp = _ensure_feedback_file(path)
    with open(fp, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "original_transcript",
                "edited_transcript",
                "remarks",
                "replacements_json",
            ],
        )
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(),
                "original_transcript": (original_transcript or "").strip(),
                "edited_transcript": (edited_transcript or "").strip(),
                "remarks": (remarks or "").strip(),
                "replacements_json": json.dumps(pairs, ensure_ascii=True),
            }
        )
    return {"feedback_path": str(fp), "replacement_pairs": pairs}


def load_feedback_phrase_map(*, path: str = DEFAULT_FEEDBACK_PATH, max_rows: int = 1000) -> Dict[str, str]:
    dsn = _feedback_db_dsn()
    if dsn:
        db_phrase_map = _load_feedback_phrase_map_postgres(
            dsn=dsn,
            table_name=DEFAULT_FEEDBACK_TABLE,
            max_rows=max_rows,
        )
        if db_phrase_map is not None:
            return db_phrase_map

    fp = Path(path)
    if not fp.exists():
        return {}

    phrase_map: Dict[str, str] = {}
    try:
        with open(fp, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx >= max_rows:
                    break
                raw = (row.get("replacements_json") or "").strip()
                if not raw:
                    continue
                try:
                    pairs = json.loads(raw)
                except Exception:
                    continue
                for pair in pairs:
                    wrong = str((pair or {}).get("wrong") or "").strip()
                    correct = str((pair or {}).get("correct") or "").strip()
                    if len(wrong) < 3 or len(correct) < 1:
                        continue
                    phrase_map[wrong.lower()] = correct
    except Exception:
        return {}
    return phrase_map
