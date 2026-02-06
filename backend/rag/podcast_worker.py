# rag/podcast_worker.py

import subprocess
from pathlib import Path
import traceback

from db import get_repo
from rag.vector_store import load_texts
from rag.podcast import generate_podcast_script
from rag.tts_client import generate_audio_segment


AUDIO_DIR = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def run_podcast_job(
    job_id: str,
    notebook_id: str,
    user_id: str,
    speakers: int,
):
    repo = get_repo()
    cur = repo.conn.cursor()

    try:
        # 1️⃣ Mark job as running
        cur.execute(
            """
            UPDATE podcast_jobs
            SET status = %s
            WHERE id = %s AND user_id = %s
            """,
            ("running", job_id, user_id),
        )
        repo.conn.commit()

        # 2️⃣ Load notebook text
        texts = load_texts(notebook_id)
        if not texts:
            raise Exception("Notebook content not found")

        context = "\n".join(texts[:15])

        # 3️⃣ Generate podcast script
        script = generate_podcast_script(
            context=context,
            speakers=int(speakers),
        )

        # 4️⃣ Save script immediately (UI depends on this)
        cur.execute(
            """
            UPDATE podcast_jobs
            SET status = %s,
                result = %s
            WHERE id = %s AND user_id = %s
            """,
            ("script_ready", script, job_id, user_id),
        )
        repo.conn.commit()

        # 5️⃣ Generate multi-speaker audio (best effort)
        try:
            speaker_map = {
                "Rahul": "rahul",
                "Priya": "priya",
                "Vikas": "vikas",
                "Anita": "anita",
            }

            segments = []

            for raw_line in script.splitlines():
                line = raw_line.strip()
                if ":" not in line:
                    continue

                speaker_name, text = line.split(":", 1)
                speaker_name = speaker_name.strip()
                text = text.strip()

                if speaker_name not in speaker_map or not text:
                    continue

                speaker_id = speaker_map[speaker_name]
                segment_path = AUDIO_DIR / f"{job_id}_{speaker_id}_{len(segments)}.wav"

                generate_audio_segment(
                    text=text,
                    speaker=speaker_id,
                    output_path=str(segment_path),
                )

                segments.append(segment_path)

            if not segments:
                raise Exception("No audio segments generated")

            # 6️⃣ Concatenate audio segments
            concat_file = AUDIO_DIR / f"{job_id}_concat.txt"
            final_audio = AUDIO_DIR / f"{job_id}.wav"

            with open(concat_file, "w") as f:
                for seg in segments:
                    f.write(f"file '{seg.resolve()}'\n")

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-c",
                    "copy",
                    str(final_audio),
                ],
                check=True,
            )

            # 7️⃣ Save final audio path
            cur.execute(
                """
                UPDATE podcast_jobs
                SET status = %s,
                    audio_path = %s
                WHERE id = %s AND user_id = %s
                """,
                ("done", str(final_audio), job_id, user_id),
            )
            repo.conn.commit()

            print(f"[PODCAST] Audio generated for job {job_id}")

        except Exception as tts_err:
            # Script is still usable even if TTS fails
            print("[PODCAST] TTS failed:", tts_err)

    except Exception as e:
        repo.conn.rollback()
        cur.execute(
            """
            UPDATE podcast_jobs
            SET status = %s,
                error = %s
            WHERE id = %s AND user_id = %s
            """,
            ("error", str(e), job_id, user_id),
        )
        repo.conn.commit()

        print("❌ Podcast job failed")
        traceback.print_exc()

