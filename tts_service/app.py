from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TTS.api import TTS
from pathlib import Path
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

        # ✅ THIS IS THE ONLY CORRECT CALL FOR XTTS
        tts.tts_to_file(
            text=req.text,
            speaker_wav=str(speaker_wav),
            language="en",
            file_path=str(output_path),
        )

        return {"audio_file": output_path.name}

    except Exception as e:
        print("❌ TTS ERROR")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
