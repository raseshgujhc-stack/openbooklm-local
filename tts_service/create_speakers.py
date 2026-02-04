from TTS.api import TTS
from pathlib import Path

# Paths
SPEAKER_ROOT = Path("/data/speakers")

# Create folders (one per speaker)
SPEAKERS = {
    "vikas": SPEAKER_ROOT / "rasesh",
    "anita": SPEAKER_ROOT / "anita",
    "rahul": SPEAKER_ROOT / "rahul",
    "priya": SPEAKER_ROOT / "priya",
}

for name, path in SPEAKERS.items():
    path.mkdir(parents=True, exist_ok=True)

print("✅ Speaker folders ensured:")
for name in SPEAKERS:
    print(f" - {name}: {SPEAKERS[name]}")

# Load XTTS
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=False
)

# OPTIONAL: quick smoke test (only if reference audio exists)
for speaker, folder in SPEAKERS.items():
    wavs = list(folder.glob("*.wav"))
    if not wavs:
        print(f"⚠️ No reference audio yet for '{speaker}' (this is OK)")
        continue

    print(f"🎙 Testing speaker '{speaker}' using {wavs[0].name}")

    tts.tts_to_file(
        text="Hi, and welcome to this special podcast episode where we're discussing the groundbreaking Vision Document released by Honourable Mr. Justice Surya Kant, Judge of the Supreme Court of India, during the inauguration program of the eGujarat High Court Portal on the 7th of December, 2024. This Vision Document, Version 1.0, is a significant step towards digital transformation in the Indian judicial system. Mrs. Justice Sunita Agarwal, Chief Justice of the High Court, expressed her gratitude to His Lordship for bestowing his patronage on this endeavor.",
        speaker_wav=f"/data/speakers/{speaker}/ref1.wav",
        language="en",
        file_path=f"/tmp/{speaker}.wav",
    )

print("✅ XTTS speaker setup complete")
