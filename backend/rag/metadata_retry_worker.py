import time
from pathlib import Path

from db import get_repo
from psycopg2.extras import Json
from rag.act_catalog import extract_acts_with_sections, get_catalog_source
from rag.ingest import normalize_and_dedup_acts
from rag.pdf_reader import read_pdf_from_path


POLL_INTERVAL = 4
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = [0, 60, 300, 1800]
INGEST_BACKLOG_PAUSE_SECONDS = 8
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"


def _derive_status(metadata_conf: float, acts_len: int):
    if acts_len > 0 and metadata_conf >= 0.85:
        return "complete", False
    if metadata_conf < 0.6:
        return "failed", True
    return "needs_review", True


def metadata_retry_worker_loop():
    print("🟢 Metadata retry worker started")

    while True:
        repo = get_repo()
        cur = repo.conn.cursor()

        cur.execute(
            """
            SELECT value
            FROM admin_runtime_settings
            WHERE key = 'metadata_retry_paused'
            """
        )
        paused_row = cur.fetchone()
        is_paused = bool(paused_row and str(paused_row[0]).lower() == "true")
        if is_paused:
            repo.conn.commit()
            time.sleep(POLL_INTERVAL)
            continue

        # Do not compete with active uploads/ingestion.
        cur.execute(
            """
            SELECT COUNT(*)
            FROM ingest_jobs
            WHERE status IN ('queued', 'processing')
            """
        )
        ingest_backlog = cur.fetchone()[0]
        if ingest_backlog and int(ingest_backlog) > 0:
            repo.conn.commit()
            time.sleep(INGEST_BACKLOG_PAUSE_SECONDS)
            continue

        cur.execute(
            """
            SELECT id, document_id, user_id, attempts
            FROM metadata_retry_jobs
            WHERE status = 'queued'
              AND next_retry_at <= NOW()
              AND attempts < %s
            ORDER BY updated_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            """,
            (MAX_ATTEMPTS,),
        )
        row = cur.fetchone()
        if not row:
            repo.conn.commit()
            time.sleep(POLL_INTERVAL)
            continue

        job_id, document_id, user_id, attempts = row
        next_attempt = int(attempts) + 1

        cur.execute(
            """
            UPDATE metadata_retry_jobs
            SET status = 'processing', attempts = attempts + 1, updated_at = NOW()
            WHERE id = %s
            """,
            (job_id,),
        )
        repo.conn.commit()

        try:
            pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
            if not pdf_path.exists():
                raise RuntimeError("PDF missing for retry")

            text = read_pdf_from_path(pdf_path)
            if not text.strip():
                raise RuntimeError("Empty PDF text on retry")

            raw_acts = extract_acts_with_sections(text)
            acts = normalize_and_dedup_acts(raw_acts)

            cur.execute(
                """
                SELECT field_confidence, extraction_notes
                FROM document_metadata
                WHERE document_id = %s
                """,
                (document_id,),
            )
            existing = cur.fetchone()
            if not existing:
                raise RuntimeError("document_metadata row missing")

            field_confidence = existing[0] or {}
            extraction_notes = existing[1] or {}
            if not isinstance(field_confidence, dict):
                field_confidence = {}
            if not isinstance(extraction_notes, dict):
                extraction_notes = {}

            field_confidence["act_names"] = 0.9 if acts else 0.0

            # Recompute a conservative metadata confidence from available footprints.
            confidence_keys = [
                "court_name",
                "case_number",
                "judge_name",
                "order_date",
                "decision_status",
                "document_type",
                "act_names",
                "primary_topics",
            ]
            present = 0
            for k in confidence_keys:
                if float(field_confidence.get(k, 0.0) or 0.0) > 0:
                    present += 1
            metadata_conf = round(present / len(confidence_keys), 2)
            extraction_status, needs_review = _derive_status(metadata_conf, len(acts))

            extraction_notes["acts_retry_source"] = f"catalog-based retry ({get_catalog_source()})"
            extraction_notes["retry_attempts"] = attempts + 1
            extraction_notes["missing_fields"] = [k for k, v in field_confidence.items() if float(v or 0) == 0.0]

            cur.execute(
                """
                UPDATE document_metadata
                SET act_names = %s,
                    referenced_laws = %s,
                    field_confidence = %s,
                    metadata_confidence = %s,
                    extraction_notes = %s,
                    extraction_status = %s,
                    needs_review = %s,
                    retry_count = COALESCE(retry_count, 0) + 1,
                    last_retry_at = NOW()
                WHERE document_id = %s
                """,
                (
                    Json(acts),
                    Json([a["act"] for a in acts]),
                    Json(field_confidence),
                    str(metadata_conf),
                    Json(extraction_notes),
                    extraction_status,
                    needs_review,
                    document_id,
                ),
            )

            final_job_status = "done" if extraction_status == "complete" else "queued"
            backoff_idx = min(next_attempt, len(BACKOFF_SECONDS) - 1)
            delay_sec = BACKOFF_SECONDS[backoff_idx]
            cur.execute(
                """
                UPDATE metadata_retry_jobs
                SET status = %s,
                    next_retry_at = NOW() + make_interval(secs => %s),
                    last_error = NULL,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (final_job_status, delay_sec, job_id),
            )
            repo.conn.commit()

        except Exception as e:
            backoff_idx = min(next_attempt, len(BACKOFF_SECONDS) - 1)
            delay_sec = BACKOFF_SECONDS[backoff_idx]
            terminal = next_attempt >= MAX_ATTEMPTS
            cur.execute(
                """
                UPDATE metadata_retry_jobs
                SET status = %s,
                    next_retry_at = NOW() + make_interval(secs => %s),
                    last_error = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                ("failed" if terminal else "queued", delay_sec, str(e), job_id),
            )
            repo.conn.commit()
            print(f"Metadata retry failed for {document_id}: {e}")
