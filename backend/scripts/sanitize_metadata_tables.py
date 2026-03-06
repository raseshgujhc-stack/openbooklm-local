#!/usr/bin/env python3
"""
Sanitize metadata tables for consistent downstream retrieval.

Actions:
1) Trim and collapse whitespace for key text columns.
2) Convert empty strings to NULL.
3) Sync canonical reference-book counts from document_metadata -> book_metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_repo


def run() -> None:
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        UPDATE document_metadata
        SET
          filename = NULLIF(regexp_replace(btrim(filename), '\\s+', ' ', 'g'), ''),
          court_name = NULLIF(regexp_replace(btrim(court_name), '\\s+', ' ', 'g'), ''),
          case_number = NULLIF(regexp_replace(btrim(case_number), '\\s+', ' ', 'g'), ''),
          case_type = NULLIF(regexp_replace(btrim(case_type), '\\s+', ' ', 'g'), ''),
          document_type = NULLIF(regexp_replace(btrim(document_type), '\\s+', ' ', 'g'), ''),
          decision_status = NULLIF(regexp_replace(btrim(decision_status), '\\s+', ' ', 'g'), ''),
          document_about = NULLIF(regexp_replace(btrim(document_about), '\\s+', ' ', 'g'), '')
        """
    )
    trim_doc = cur.rowcount

    cur.execute(
        """
        UPDATE book_metadata
        SET
          filename = NULLIF(regexp_replace(btrim(filename), '\\s+', ' ', 'g'), ''),
          title = NULLIF(regexp_replace(btrim(title), '\\s+', ' ', 'g'), ''),
          language = NULLIF(regexp_replace(btrim(language), '\\s+', ' ', 'g'), ''),
          source_type = NULLIF(regexp_replace(btrim(source_type), '\\s+', ' ', 'g'), '')
        """
    )
    trim_book = cur.rowcount

    cur.execute(
        """
        UPDATE book_metadata bm
        SET
          page_count = dm.page_count,
          word_count = dm.word_count,
          filename = COALESCE(NULLIF(bm.filename, ''), dm.filename),
          user_id = COALESCE(bm.user_id, dm.user_id),
          collection_id = COALESCE(bm.collection_id, dm.collection_id)
        FROM document_metadata dm
        WHERE dm.document_id = bm.document_id
          AND dm.document_role = 'ReferenceBook'
          AND (
            bm.page_count IS DISTINCT FROM dm.page_count
            OR bm.word_count IS DISTINCT FROM dm.word_count
            OR bm.filename IS DISTINCT FROM dm.filename
            OR bm.user_id IS DISTINCT FROM dm.user_id
            OR bm.collection_id IS DISTINCT FROM dm.collection_id
          )
        """
    )
    sync_book = cur.rowcount

    cur.execute(
        """
        SELECT COUNT(*)
        FROM document_metadata dm
        JOIN book_metadata bm ON bm.document_id = dm.document_id
        WHERE dm.document_role = 'ReferenceBook'
          AND (
            bm.page_count IS DISTINCT FROM dm.page_count
            OR bm.word_count IS DISTINCT FROM dm.word_count
          )
        """
    )
    remaining_mismatch = int((cur.fetchone() or [0])[0] or 0)

    repo.conn.commit()

    print(
        {
            "trim_document_metadata_rows": trim_doc,
            "trim_book_metadata_rows": trim_book,
            "synced_reference_books": sync_book,
            "remaining_page_word_mismatches": remaining_mismatch,
        }
    )


if __name__ == "__main__":
    run()
