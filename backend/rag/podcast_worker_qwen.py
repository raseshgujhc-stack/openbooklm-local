import subprocess
import traceback
from pathlib import Path
import re
from typing import List, Optional, Tuple

from db import get_repo
from rag.model_router import qwen_podcast_script
from rag.script_rectifier import normalize_and_validate_podcast_script
from rag.tts_client import generate_audio_segment
from rag.vector_store import load_texts

AUDIO_DIR = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

SPEAKER_NAMES = ["Rahul", "Priya", "Vikas", "Anita"]
_ENDING_STOPWORDS = {
    "and",
    "or",
    "of",
    "to",
    "for",
    "with",
    "by",
    "from",
    "as",
    "in",
    "on",
    "at",
    "that",
    "which",
    "who",
    "whom",
    "whose",
}


def _clean_podcast_context(raw: str) -> str:
    """
    Remove common legal PDF footer/header artifacts before prompting Qwen.
    This avoids dialogue contamination with lines like page counters, upload stamps,
    neutral citation tails, and downloaded-on metadata.
    """
    lines = [ln.strip() for ln in (raw or "").splitlines()]
    cleaned = []
    seen = set()

    noise_patterns = [
        r"^page\s+\d+\s+of\s+\d+\b",
        r"^---\s*page\s+\d+\s*---$",
        r"\bdownloaded on\b",
        r"\buploaded by\b",
        r"\bneutral citation\b",
        r"^\d{4}:gujhc:\d+\b",
        r"\border dated\b",
        r"^[rc]/[a-z0-9./-]+$",
    ]

    compiled = [re.compile(p, flags=re.IGNORECASE) for p in noise_patterns]

    for line in lines:
        if not line:
            continue
        if any(p.search(line) for p in compiled):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(line)

    return "\n".join(cleaned)


def _build_podcast_prompt(context: str, speakers: int, user_id: Optional[str]) -> str:
    speakers = max(1, min(speakers, 4))
    active = SPEAKER_NAMES[:speakers]
    speaker_rules = "\n".join([f"- {name}: speaks naturally" for name in active])
    style_hints = _feedback_style_hints_for_user(user_id=user_id)

    return f"""
You are a professional Indian legal podcast script writer.

STRICT RULES:
- Output ONLY dialogue lines
- EACH line MUST start with one of these speaker names EXACTLY: {", ".join(active)}
- Format: SpeakerName: sentence
- Do NOT use 'Speaker 1/2'
- Do NOT include headings, bullets, stage directions, or blank lines
- Ignore PDF artifacts like page counters, neutral citation stamps, upload/download metadata
- Keep legal facts accurate to provided content
- Use clear conversational English
- Prefer short, direct turns over long monologues

Allowed speakers:
{speaker_rules}

Editing feedback from accepted user corrections:
{style_hints}

Source content:
{context}

Generate the podcast script now.
"""


def _last_dialogue_line(script: str, allowed_names: List[str]) -> str:
    escaped = "|".join(re.escape(name) for name in allowed_names)
    pat = re.compile(rf"^\s*(?:{escaped})\s*:\s*.+$", flags=re.MULTILINE)
    matches = [m.group(0).strip() for m in pat.finditer(script or "")]
    return matches[-1] if matches else ""


def _script_looks_truncated(script: str, allowed_names: List[str]) -> bool:
    last_line = _last_dialogue_line(script, allowed_names)
    if not last_line:
        return True
    content = last_line.split(":", 1)[1].strip() if ":" in last_line else last_line
    if not content:
        return True
    if content.endswith((".", "!", "?", "\"", ".'", ".'\"", ".'”")):
        return False
    last_word = re.sub(r"[^A-Za-z]+", "", content.split()[-1]).lower() if content.split() else ""
    if last_word in _ENDING_STOPWORDS:
        return True
    # Also treat very short trailing fragments as suspect.
    if len(content.split()) < 5:
        return True
    return False


def _continue_truncated_script(script: str, speakers: int) -> str:
    active = SPEAKER_NAMES[: max(1, min(int(speakers or 2), 4))]
    prompt = f"""
You are continuing a partially generated legal podcast script.

STRICT RULES:
- Continue from where the draft stopped.
- Output ONLY dialogue lines in format: SpeakerName: sentence
- Allowed speakers ONLY: {", ".join(active)}
- Complete the interrupted thought naturally in the first continued line.
- Add 4 to 8 additional complete dialogue lines.
- Do NOT repeat full earlier lines verbatim.
- End with a complete sentence.

Current draft:
{script}

Continue now:
"""
    cont = qwen_podcast_script(prompt=prompt, max_tokens=450)
    return (cont or "").strip()


def _feedback_style_hints_for_user(user_id: Optional[str], limit: int = 3) -> str:
    if not user_id:
        return "- No prior feedback yet."
    try:
        repo = get_repo()
        cur = repo.conn.cursor()
        cur.execute(
            """
            SELECT original_script, edited_script, remarks
            FROM podcast_script_feedback
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        rows = cur.fetchall() or []
        if not rows:
            return "- No prior feedback yet."

        hints = []
        for idx, row in enumerate(rows, start=1):
            original = (row[0] or "").strip().splitlines()[:1]
            edited = (row[1] or "").strip().splitlines()[:1]
            remark = (row[2] or "").strip()
            before = original[0] if original else ""
            after = edited[0] if edited else ""
            hints.append(
                f"- Example {idx}: prefer `{after[:120]}` over `{before[:120]}`"
                + (f" (remark: {remark[:120]})" if remark else "")
            )
        return "\n".join(hints)
    except Exception:
        return "- No prior feedback yet."


def _extract_speaker_turns(script: str, allowed_names: List[str]) -> List[Tuple[str, str]]:
    escaped = "|".join(re.escape(name) for name in allowed_names)
    pattern = re.compile(rf"^(?P<speaker>{escaped})\s*:\s*", flags=re.MULTILINE)
    matches = list(pattern.finditer(script or ""))
    turns: List[Tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(script)
        text = (script[start:end] or "").strip()
        if not text:
            continue
        turns.append((match.group("speaker"), _table_markdown_to_speech(text)))
    return turns


def _table_markdown_to_speech(text: str) -> str:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    out = []
    table_rows = []

    def flush_table():
        if not table_rows:
            return
        out.append("Table summary.")
        for idx, row in enumerate(table_rows, start=1):
            if re.fullmatch(r"\s*\|?[\-\s|:]+\|?\s*", row):
                continue
            cells = [c.strip() for c in row.strip("|").split("|")]
            cells = [c for c in cells if c]
            if cells:
                out.append(f"Row {idx}: " + ", ".join(cells) + ".")
        table_rows.clear()

    for line in lines:
        if "|" in line:
            table_rows.append(line)
            continue
        flush_table()
        if line.strip():
            out.append(line.strip())

    flush_table()
    return " ".join(out).strip()


def synthesize_podcast_audio(job_id: str, user_id: str, script: str, speakers: int):
    repo = get_repo()
    cur = repo.conn.cursor()

    try:
        speaker_map = {
            "Rahul": "rahul",
            "Priya": "priya",
            "Vikas": "vikas",
            "Anita": "anita",
        }
        active_speakers = SPEAKER_NAMES[: max(1, min(int(speakers or 2), 4))]
        turns = _extract_speaker_turns(script=script, allowed_names=active_speakers)
        if not turns:
            raise Exception("No valid speaker turns found for TTS.")

        segments = []
        for speaker_name, text in turns:
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
            raise Exception("No audio segments generated.")

        concat_file = AUDIO_DIR / f"{job_id}_concat.txt"
        final_audio = AUDIO_DIR / f"{job_id}.wav"
        with open(concat_file, "w", encoding="utf-8") as f:
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

        cur.execute(
            """
            UPDATE podcast_jobs
            SET status = %s, audio_path = %s, error = NULL, status_updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
            """,
            ("done", str(final_audio), job_id, user_id),
        )
        repo.conn.commit()
    except Exception as err:
        repo.conn.rollback()
        cur.execute(
            """
            UPDATE podcast_jobs
            SET status = %s, error = %s, status_updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
            """,
            ("error", f"TTS failed: {err}", job_id, user_id),
        )
        repo.conn.commit()


def run_podcast_job_qwen(
    job_id: str,
    notebook_id: str,
    user_id: str,
    speakers: int,
    auto_generate_audio: bool = False,
):
    repo = get_repo()
    cur = repo.conn.cursor()

    try:
        cur.execute(
            """
            UPDATE podcast_jobs
            SET status = %s, status_updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
            """,
            ("running", job_id, user_id),
        )
        repo.conn.commit()

        texts = load_texts(notebook_id)
        if not texts:
            raise Exception("Notebook content not found")

        raw_context = "\n".join(texts[:20])
        context = _clean_podcast_context(raw_context)
        if not context.strip():
            raise Exception("Notebook content is empty after cleaning")
        prompt = _build_podcast_prompt(
            context=context,
            speakers=int(speakers),
            user_id=user_id,
        )
        script = qwen_podcast_script(prompt=prompt, max_tokens=1800)
        active_speakers = SPEAKER_NAMES[: max(1, min(int(speakers or 2), 4))]
        if _script_looks_truncated(script, active_speakers):
            continuation = _continue_truncated_script(script=script, speakers=speakers)
            if continuation:
                script = f"{script.rstrip()}\n{continuation.lstrip()}"
        script, _ = normalize_and_validate_podcast_script(script=script, speakers=speakers)

        cur.execute(
            """
            UPDATE podcast_jobs
            SET status = %s, result = %s, error = NULL, status_updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
            """,
            ("script_ready", script, job_id, user_id),
        )
        repo.conn.commit()

        if auto_generate_audio:
            cur.execute(
                """
                UPDATE podcast_jobs
                SET status = %s, status_updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND user_id = %s
                """,
                ("running", job_id, user_id),
            )
            repo.conn.commit()
            synthesize_podcast_audio(
                job_id=job_id,
                user_id=user_id,
                script=script,
                speakers=speakers,
            )

    except Exception as e:
        repo.conn.rollback()
        cur.execute(
            """
            UPDATE podcast_jobs
            SET status = %s, error = %s, status_updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
            """,
            ("error", str(e), job_id, user_id),
        )
        repo.conn.commit()
        print("Podcast Qwen job failed")
        traceback.print_exc()
