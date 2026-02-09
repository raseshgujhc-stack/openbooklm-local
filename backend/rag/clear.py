# rag/tts_client.py
import requests

TTS_BASE = "http://localhost:9000"

def generate_audio_segment(text: str, speaker: str, output_path: str):
    res = requests.post(
        f"{TTS_BASE}/tts",
        json={
            "text": text,
            "speaker": speaker,
            "output_path": output_path,
        },
        timeout=300,
    )

    res.raise_for_status()
    return output_path
