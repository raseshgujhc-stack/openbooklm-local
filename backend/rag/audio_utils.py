# rag/audio_utils.py
import wave
from pathlib import Path

def merge_wav_files(wav_files, output_path):
    if not wav_files:
        return

    with wave.open(str(wav_files[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]

    for wav in wav_files[1:]:
        with wave.open(str(wav), "rb") as w:
            frames.append(w.readframes(w.getnframes()))

    with wave.open(str(output_path), "wb") as out:
        out.setparams(params)
        for f in frames:
            out.writeframes(f)
