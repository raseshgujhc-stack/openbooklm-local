#!/usr/bin/env python3
"""
Backfill document_metadata quality fields for already-ingested rows.

Repairs:
- page_count (NULL/<=0) from actual uploaded PDF page count, fallback to estimate
- word_count (NULL/<=0) from stored FAISS chunk text
- domain_confidence (NULL) from role/domain
- field_confidence / metadata_confidence (NULL) with deterministic scoring
"""

from __future__ import annotations

import json
from pathlib import Path
import argparse

import psycopg2
from psycopg2.extras import Json
from pypdf import PdfReader
from dotenv import load_dotenv
import os


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
UPLOAD_DIR = ROOT / "data" / "uploads"
FAISS_DIR = ROOT / "data" / "faiss"


def _count_pdf_pages(pdf_path: Path) -> int | None:
    try:
        with open(pdf_path, "rb") as f:
            return len(PdfReader(f).pages)
    except Exception:
        return None


def _word_count_from_faiss_json(doc_id: str) -> int | None:
    meta_path = FAISS_DIR / f"{doc_id}.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        text = "\n".join((row.get("text") or "") for row in data if isinstance(row, dict))
        words = len(text.split())
        return words if words > 0 else None
    except Exception:
        return None


def _estimate_pages_from_words(word_count: int | None) -> int | None:
    if not word_count or word_count <= 0:
        return None
    return max(1, round(word_count / 800))


def _compute_confidence(row: dict) -> tuple[float, dict]:
    confidence_fields = {
        "court_name": row.get("court_name"),
        "case_number": row.get("case_number"),
        "judge_name": row.get("judge_name"),
        "order_date": row.get("order_date"),
        "decision_status": row.get("decision_status"),
        "document_type": row.get("document_type"),
        "act_names": row.get("act_names"),
        "primary_topics": row.get("primary_topics"),
    }
    filled = 0
    field_conf = {}
    for key, value in confidence_fields.items():
        ok = bool(value) if not isinstance(value, list) else len(value) > 0
        field_conf[key] = 0.9 if ok else 0.0
        if ok:
            filled += 1
    return round(filled / len(confidence_fields), 2), field_conf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", help="Backfill only one document_id")
    args = parser.parse_args()

    load_dotenv(ENV_PATH)
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("POSTGRES_DSN not set in backend/.env")

    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    if args.document_id:
        cur.execute(
            """
            SELECT
                document_id,
                page_count,
                word_count,
                document_role,
                domain,
                court_name,
                case_number,
                judge_name,
                order_date,
                decision_status,
                document_type,
                act_names,
                primary_topics,
                metadata_confidence,
                field_confidence,
                domain_confidence,
                extraction_notes
            FROM document_metadata
            WHERE document_id = %s
            """,
            (args.document_id,),
        )
    else:
        # Intentionally broad: page_count mismatches are only detectable
        # after checking the underlying PDF bytes on disk.
        cur.execute(
            """
            SELECT
                document_id,
                page_count,
                word_count,
                document_role,
                domain,
                court_name,
                case_number,
                judge_name,
                order_date,
                decision_status,
                document_type,
                act_names,
                primary_topics,
                metadata_confidence,
                field_confidence,
                domain_confidence,
                extraction_notes
            FROM document_metadata
            """
        )

    rows = cur.fetchall()
    if not rows:
        print("No rows require backfill.")
        conn.close()
        return

    updated = 0
    for r in rows:
        row = {
            "document_id": r[0],
            "page_count": r[1],
            "word_count": r[2],
            "document_role": r[3],
            "domain": r[4],
            "court_name": r[5],
            "case_number": r[6],
            "judge_name": r[7],
            "order_date": r[8],
            "decision_status": r[9],
            "document_type": r[10],
            "act_names": r[11],
            "primary_topics": r[12],
            "metadata_confidence": r[13],
            "field_confidence": r[14],
            "domain_confidence": r[15],
            "extraction_notes": r[16],
        }

        doc_id = row["document_id"]
        new_word_count = row["word_count"]
        if not new_word_count or new_word_count <= 0:
            new_word_count = _word_count_from_faiss_json(doc_id)

        new_page_count = row["page_count"]
        pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
        actual_pdf_pages = _count_pdf_pages(pdf_path) if pdf_path.exists() else None
        if actual_pdf_pages:
            # Prefer authoritative file page count even if current value is 1/non-null.
            new_page_count = actual_pdf_pages
        elif not new_page_count or new_page_count <= 0:
            new_page_count = _estimate_pages_from_words(new_word_count)

        if row["domain_confidence"] is None:
            role = (row["document_role"] or "").lower()
            domain = (row["domain"] or "").lower()
            if role == "judicial" or domain == "judicial":
                new_domain_confidence = 0.9
            else:
                new_domain_confidence = 0.6
        else:
            new_domain_confidence = row["domain_confidence"]

        if row["metadata_confidence"] is None or row["field_confidence"] is None:
            m_conf, f_conf = _compute_confidence(row)
        else:
            m_conf, f_conf = row["metadata_confidence"], row["field_confidence"]

        notes = row["extraction_notes"] or {}
        if not isinstance(notes, dict):
            notes = {}
        notes["backfill"] = "counts/confidence repaired by script"

        changed = (
            new_page_count != row["page_count"]
            or new_word_count != row["word_count"]
            or new_domain_confidence != row["domain_confidence"]
            or m_conf != row["metadata_confidence"]
            or f_conf != row["field_confidence"]
        )
        if not changed:
            continue

        cur.execute(
            """
            UPDATE document_metadata
            SET
                page_count = %s,
                word_count = %s,
                domain_confidence = %s,
                metadata_confidence = %s,
                field_confidence = %s,
                extraction_notes = %s
            WHERE document_id = %s
            """,
            (
                new_page_count,
                new_word_count,
                new_domain_confidence,
                m_conf,
                Json(f_conf) if isinstance(f_conf, dict) else f_conf,
                Json(notes),
                doc_id,
            ),
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"Updated {updated} document_metadata rows.")


if __name__ == "__main__":
    main()
