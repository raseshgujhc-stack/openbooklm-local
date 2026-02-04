import base64
import subprocess
import tempfile
import numpy as np
import whisper
import soundfile as sf
from datetime import datetime

from core_models.legal_formatter import LegalFormatter


class JudicialTranscriber:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None
        self.legal_formatter = LegalFormatter()

    def load_model(self):
        if self.model is None:
            print(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(
                self.model_size,
                device="cpu"
            )
        return self.model

    def _decode_webm_to_pcm(self, audio_base64: str) -> np.ndarray:
        """Decode WebM/Opus base64 audio to 16kHz mono PCM float32"""
        audio_bytes = base64.b64decode(audio_base64)

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
            webm_file.write(audio_bytes)
            webm_path = webm_file.name

        wav_path = webm_path.replace(".webm", ".wav")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", webm_path,
                "-ac", "1",
                "-ar", "16000",
                wav_path
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        audio, _ = sf.read(wav_path, dtype="float32")
        return audio

    def transcribe_chunk(self, audio_base64: str, language: str = "en"):
        try:
            audio_np = self._decode_webm_to_pcm(audio_base64)
            model = self.load_model()

            result = model.transcribe(
                audio_np,
                language=language,
                fp16=False,
                initial_prompt="Indian court dictation. Legal terminology. Judicial proceeding."
            )

            text = result["text"].strip()
            formatted = self.legal_formatter.format_realtime(text)

            return {
                "text": text,
                "formatted": formatted,
                "language": result.get("language", language),
                "confidence": 1.0,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "text": "",
                "formatted": "",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def transcribe_file(self, audio_path: str, language: str = "en"):
        model = self.load_model()

        result = model.transcribe(
            audio_path,
            language=language,
            fp16=False,
            initial_prompt="Indian court dictation. Legal terminology. Judicial proceeding."
        )

        text = result["text"].strip()
        formatted = self.legal_formatter.format_complete_document(
            text,
            chunks=[],
            filename=audio_path
        )

        import librosa
        duration = librosa.get_duration(path=audio_path)

        return {
            "text": text,
            "formatted": formatted,
            "language": result.get("language", language),
            "confidence": 1.0,
            "duration": duration
        }

