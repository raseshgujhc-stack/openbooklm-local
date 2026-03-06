"""
TTS microservice for podcast/audio generation.

Exposes:
- /health for readiness checks
- /tts to synthesize text into WAV using XTTS v2 and speaker reference audio
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TTS.api import TTS
from pathlib import Path
import tempfile
import wave
import re
import traceback

app = FastAPI()

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=False,
)

AUDIO_DIR = Path("/app/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

SPEAKER_DIR = Path("/data/speakers")


class TTSRequest(BaseModel):
    text: str
    speaker: str
    output_path: str


MAX_TTS_CHARS = 220
CHUNK_SILENCE_MS = 90


def _split_long_text(text: str, max_chars: int = MAX_TTS_CHARS) -> list[str]:
    """
    Split long text into XTTS-safe chunks to avoid truncation from model limits.
    """
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    # First pass: sentence-aware split.
    sentences = re.split(r"(?<=[.!?;:])\s+", cleaned)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        if len(s) > max_chars:
            # Second pass: hard word wrap for oversized sentence.
            words = s.split()
            piece = ""
            for w in words:
                candidate = (piece + " " + w).strip()
                if len(candidate) <= max_chars:
                    piece = candidate
                else:
                    if piece:
                        chunks.append(piece)
                    piece = w
            if piece:
                chunks.append(piece)
            continue
        candidate = (current + " " + s).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


def _concat_wav_files(input_paths: list[Path], output_path: Path, silence_ms: int = CHUNK_SILENCE_MS) -> None:
    """
    Concatenate mono WAV chunk files and inject tiny silence between chunks so
    transitions sound natural on long synthesis.
    """
    if not input_paths:
        raise ValueError("No input wav files to concatenate")

    with wave.open(str(input_paths[0]), "rb") as first:
        nchannels = first.getnchannels()
        sampwidth = first.getsampwidth()
        framerate = first.getframerate()
        comptype = first.getcomptype()
        compname = first.getcompname()

    silence_frames = int((framerate * max(0, silence_ms)) / 1000.0)
    silence_bytes = b"\x00" * (silence_frames * sampwidth * nchannels)

    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(nchannels)
        out.setsampwidth(sampwidth)
        out.setframerate(framerate)
        out.setcomptype(comptype, compname)

        for idx, path in enumerate(input_paths):
            with wave.open(str(path), "rb") as src:
                if (
                    src.getnchannels() != nchannels
                    or src.getsampwidth() != sampwidth
                    or src.getframerate() != framerate
                ):
                    raise ValueError("Incompatible WAV chunks during concat")
                out.writeframes(src.readframes(src.getnframes()))
            if idx < len(input_paths) - 1 and silence_frames > 0:
                out.writeframes(silence_bytes)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/tts")
def tts_endpoint(req: TTSRequest):
    try:
        speaker_dir = SPEAKER_DIR / req.speaker
        speaker_wav = speaker_dir / "ref1.wav"

        if not speaker_dir.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Speaker folder missing: {speaker_dir}",
            )

        if not speaker_wav.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Speaker wav missing: {speaker_wav}",
            )

        output_path = Path(req.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        chunks = _split_long_text(req.text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Text is empty after normalization")

        if len(chunks) == 1:
            tts.tts_to_file(
                text=chunks[0],
                speaker_wav=str(speaker_wav),
                language="en",
                file_path=str(output_path),
            )
        else:
            chunk_paths: list[Path] = []
            with tempfile.TemporaryDirectory(prefix="tts_chunks_") as tmpdir:
                tmp = Path(tmpdir)
                for idx, chunk_text in enumerate(chunks):
                    part_path = tmp / f"chunk_{idx:03d}.wav"
                    tts.tts_to_file(
                        text=chunk_text,
                        speaker_wav=str(speaker_wav),
                        language="en",
                        file_path=str(part_path),
                    )
                    chunk_paths.append(part_path)
                _concat_wav_files(chunk_paths, output_path)

        return {
            "audio_file": output_path.name,
            "chunk_count": len(chunks),
            "chunking_applied": len(chunks) > 1,
        }

    except Exception as e:
        print("❌ TTS ERROR")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
