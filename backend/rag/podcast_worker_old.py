import threading
import requests
from pathlib import Path

from rag.db import get_db
from rag.vector_store import load_texts
from rag.podcast import generate_podcast_script

# ---------------- CONFIG ----------------
TTS_URL = "http://localhost:9000/tts"

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def run_podcast_job(
    job_id: str,
    notebook_id: str,
    user_id: str,
    speakers: int,
):
    db = get_db()

    try:
        # 1️⃣ Mark running
        db.execute(
            "UPDATE podcast_jobs SET status=? WHERE id=?",
            ("running", job_id),
        )
        db.commit()

        # 2️⃣ Load notebook content
        texts = load_texts(notebook_id)
        if not texts:
            raise Exception("Notebook content not found")

        context = "\n".join(texts[:15])  # token safety

        # 3️⃣ Generate script (this ALWAYS works)
        script = generate_podcast_script(
            context=context,
            speakers=int(speakers),
        )

        # 4️⃣ Save script immediately
        db.execute(
            """
            UPDATE podcast_jobs
            SET status=?, result=?
            WHERE id=?
            """,
            ("script_ready", script, job_id),
        )
        db.commit()

        # 5️⃣ TTS (best-effort)
        try:
            audio_path = (AUDIO_DIR / f"{job_id}.wav").resolve()


            res = requests.post(
                TTS_URL,
                json={"text": script},
                timeout=300,
            )

            if res.ok:
                audio_path.write_bytes(res.content)

                db.execute(
                    """
                    UPDATE podcast_jobs
                    SET status=?, audio_path=?
                    WHERE id=?
                    """,
                    ("done", str(audio_path), job_id),
                )
                db.commit()

        except Exception as tts_error:
            # Script remains usable
            print("TTS failed:", tts_error)

    except Exception as e:
        db.execute(
            "UPDATE podcast_jobs SET status=?, error=? WHERE id=?",
            ("error", str(e), job_id),
        )
        db.commit()

    finally:
        db.close()
