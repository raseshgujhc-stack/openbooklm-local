from TTS.api import TTS
import json

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

speakers = {
    "aarav": tts.get_speaker_embedding(),
    "neha": tts.get_speaker_embedding(),
    "rohit": tts.get_speaker_embedding(),
    "kavya": tts.get_speaker_embedding(),
}

with open("speaker_embeddings.json", "w") as f:
    json.dump(
        {k: v.tolist() for k, v in speakers.items()},
        f,
        indent=2
    )

print("✅ speaker_embeddings.json created")
