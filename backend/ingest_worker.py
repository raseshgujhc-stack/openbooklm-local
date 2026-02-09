import time
from db import get_repo
from ingest_runner import run_ingestion


POLL_INTERVAL = 2  # seconds


def ingest_worker_loop():
    print("🟢 Ingest worker started")

    while True:
        repo = get_repo()
        cur = repo.conn.cursor()

        # Pick next queued job safely
        cur.execute(
            """
            SELECT
                j.job_id,
                j.notebook_id,
                n.user_id,
                n.collection_id,
                n.filename
            FROM ingest_jobs j
            JOIN notebooks n ON n.notebook_id = j.notebook_id
            WHERE j.status = 'queued'
            ORDER BY j.created_at
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            """
        )

        row = cur.fetchone()
        if not row:
            repo.conn.commit()
            time.sleep(POLL_INTERVAL)
            continue

        job_id, notebook_id, user_id, collection_id, filename = row

        # Mark processing
        cur.execute(
            """
            UPDATE ingest_jobs
            SET status='processing', started_at=now()
            WHERE job_id=%s
            """,
            (job_id,),
        )
        cur.execute(
            """
            UPDATE notebooks
            SET status='processing'
            WHERE notebook_id=%s
            """,
            (notebook_id,),
        )
        repo.conn.commit()

        try:
            run_ingestion(
                notebook_id=notebook_id,
                user_id=user_id,
                collection_id=collection_id,
                filename=filename,
            )

            cur.execute(
                """
                UPDATE ingest_jobs
                SET status='done', finished_at=now()
                WHERE job_id=%s
                """,
                (job_id,),
            )
            cur.execute(
                """
                UPDATE notebooks
                SET status='ready'
                WHERE notebook_id=%s
                """,
                (notebook_id,),
            )
            repo.conn.commit()

            print(f"✅ Ingested notebook {notebook_id}")

        except Exception as e:
            cur.execute(
                """
                UPDATE ingest_jobs
                SET status='failed', error=%s
                WHERE job_id=%s
                """,
                (str(e), job_id),
            )
            cur.execute(
                """
                UPDATE notebooks
                SET status='failed'
                WHERE notebook_id=%s
                """,
                (notebook_id,),
            )
            repo.conn.commit()

            print(f"❌ Failed ingest {notebook_id}: {e}")

