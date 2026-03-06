#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
UPLOAD_DIR = ROOT / "data" / "uploads"
sys.path.insert(0, str(ROOT))

from rag.act_catalog import extract_acts_with_sections, get_catalog_source
from rag.pdf_reader import read_pdf_from_path


def _recompute_field_confidence(existing: dict | None, acts: list[dict]) -> dict:
    existing = existing or {}
    if not isinstance(existing, dict):
        existing = {}
    existing["act_names"] = 0.9 if acts else 0.0
    existing["primary_topics"] = existing.get("primary_topics", 0.0)
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", help="Single document_id to backfill")
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
            SELECT document_id, act_names, field_confidence, extraction_notes
            FROM document_metadata
            WHERE document_id = %s
            """,
            (args.document_id,),
        )
    else:
        cur.execute(
            """
            SELECT document_id, act_names, field_confidence, extraction_notes
            FROM document_metadata
            WHERE act_names IS NULL OR act_names = '[]'::jsonb
            """
        )

    rows = cur.fetchall()
    if not rows:
        print("No rows found for acts backfill.")
        conn.close()
        return

    updated = 0
    source = get_catalog_source()

    for document_id, act_names, field_confidence, extraction_notes in rows:
        pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
        if not pdf_path.exists():
            continue

        text = read_pdf_from_path(pdf_path)
        acts = extract_acts_with_sections(text)
        if not acts:
            continue

        notes = extraction_notes or {}
        if not isinstance(notes, dict):
            notes = {}
        notes["acts_backfill"] = f"catalog-based extraction ({source})"

        new_field_conf = _recompute_field_confidence(field_confidence, acts)
        referenced_laws = [a["act"] for a in acts]

        cur.execute(
            """
            UPDATE document_metadata
            SET act_names = %s,
                referenced_laws = %s,
                field_confidence = %s,
                extraction_notes = %s
            WHERE document_id = %s
            """,
            (
                Json(acts),
                Json(referenced_laws),
                Json(new_field_conf),
                Json(notes),
                document_id,
            ),
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"Updated act_names for {updated} rows.")


if __name__ == "__main__":
    main()
