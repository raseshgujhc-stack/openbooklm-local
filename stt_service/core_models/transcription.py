"""
Whisper-based transcription service layer for STT.

Decodes WebM chunks, runs inference, and applies legal formatter output for
real-time and file-based judicial transcription flows.
"""

import base64
import subprocess
import tempfile
import os
import threading
import numpy as np
import whisper
import soundfile as sf
from datetime import datetime

from core_models.legal_formatter import LegalFormatter


class JudicialTranscriber:
    def __init__(self, model_size="base"):
        # Remark: honor env model override so quality can be tuned without code change.
        self.model_size = os.getenv("MODEL_SIZE", model_size)
        self.model = None
        self.legal_formatter = LegalFormatter()
        # Remark: decoding knobs are env-driven so quality/speed can be tuned per machine.
        self.beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
        self.best_of = int(os.getenv("WHISPER_BEST_OF", "5"))
        self.no_speech_threshold = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.45"))
        self.logprob_threshold = float(os.getenv("WHISPER_LOGPROB_THRESHOLD", "-1.0"))
        self.compression_ratio_threshold = float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4"))
        self.temperature = tuple(
            float(t.strip())
            for t in os.getenv("WHISPER_TEMPERATURES", "0.0,0.2").split(",")
            if t.strip()
        )
        if not self.temperature:
            self.temperature = (0.0, 0.2)
        self.primary_temperature = float(self.temperature[0])

    def load_model(self):
        if self.model is None:
            print(f"Loading Whisper model: {self.model_size}")
            download_root = os.getenv("WHISPER_MODEL_PATH", "/app/models/whisper")
            os.makedirs(download_root, exist_ok=True)
            self.model = whisper.load_model(
                self.model_size,
                device="cpu",
                download_root=download_root,
            )
        return self.model

    def _transcribe(self, source, language: str | None = None):
        model = self.load_model()
        kwargs = {
            "fp16": False,
            "initial_prompt": (
                "Indian court dictation. Legal terminology. "
                "Use terms: oral order, condonation of delay, returnable forthwith, "
                "learned Additional Public Prosecutor waives service for the respondent-State."
            ),
            # Remark: stable decode profile for Whisper-large on CPU.
            # Avoid mixed beam/sampling fallback tensors that can mismatch.
            "temperature": self.primary_temperature,
            "condition_on_previous_text": False,
            "no_speech_threshold": self.no_speech_threshold,
            "logprob_threshold": self.logprob_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold,
        }
        if self.primary_temperature <= 0.0:
            kwargs["beam_size"] = max(1, self.beam_size)
            kwargs["best_of"] = 1
        else:
            kwargs["best_of"] = max(1, self.best_of)
        if language:
            kwargs["language"] = language
        try:
            return model.transcribe(source, **kwargs)
        except RuntimeError as e:
            # Remark: recover from occasional Whisper tensor shape mismatch by retrying
            # with the most conservative greedy decode parameters.
            if "Sizes of tensors must match" not in str(e):
                raise
            safe_kwargs = {
                "fp16": False,
                "temperature": 0.0,
                "beam_size": 1,
                "best_of": 1,
                "condition_on_previous_text": False,
                "initial_prompt": kwargs["initial_prompt"],
            }
            if language:
                safe_kwargs["language"] = language
            return model.transcribe(source, **safe_kwargs)

    def _decode_audio_to_pcm(self, audio_base64: str, mime_type: str | None = None) -> np.ndarray:
        """Decode base64 browser audio chunk to 16kHz mono PCM float32."""
        audio_bytes = base64.b64decode(audio_base64)

        ext_map = {
            "audio/webm": ".webm",
            "audio/webm;codecs=opus": ".webm",
            "audio/ogg": ".ogg",
            "audio/mp4": ".mp4",
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
        }
        suffix = ext_map.get((mime_type or "").lower(), ".webm")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as input_file:
            input_file.write(audio_bytes)
            input_path = input_file.name

        wav_path = input_path + ".wav"

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", input_path,
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
        finally:
            for path in (input_path, wav_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass


    def transcribe_chunk(
        self,
        audio_base64: str,
        language: str = "en",
        format_type: str = "high_court",
        mime_type: str | None = None,
    ):
        try:
            audio_np = self._decode_audio_to_pcm(audio_base64, mime_type=mime_type)
            result = self._transcribe(audio_np, language=language)

            text = result["text"].strip()
            formatted = self.legal_formatter.format_realtime(text, court_type=format_type)

            return {
                "text": text,
                "formatted": formatted,
                "format_type": format_type,
                "language": result.get("language", language),
                "confidence": 1.0,
                "timestamp": datetime.now().isoformat(),
                "mime_type": mime_type,
            }

        except Exception as e:
            return {
                "text": "",
                "formatted": "",
                "format_type": format_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def transcribe_file(
        self,
        audio_path: str,
        language: str = "en",
        format_type: str | None = None,
        source_filename: str | None = None,
    ):
        normalized_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        boosted_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            # Normalize browser-recorded formats (webm/ogg/mp4/...) into stable WAV for Whisper.
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", audio_path,
                    "-ac", "1",
                    "-ar", "16000",
                    normalized_wav,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Diagnostic signal stats from normalized audio
            audio_np, sample_rate = sf.read(normalized_wav, dtype="float32")
            if audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=1)
            duration = float(len(audio_np) / sample_rate) if sample_rate else 0.0
            rms = float(np.sqrt(np.mean(np.square(audio_np)))) if len(audio_np) > 0 else 0.0
            peak = float(np.max(np.abs(audio_np))) if len(audio_np) > 0 else 0.0

            if duration < 0.2:
                raise RuntimeError(
                    f"Captured audio is too short (duration={duration:.2f}s). "
                    "Record longer and retry."
                )

            # First attempt: use requested language
            result = self._transcribe(normalized_wav, language=language)
            text = (result.get("text") or "").strip()

            # Second attempt: auto language detection
            if len(text) < 3:
                result = self._transcribe(normalized_wav, language=None)
                text = (result.get("text") or "").strip()

            # Third attempt: boost low-volume speech and retry
            if len(text) < 3:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", normalized_wav,
                        "-af", "highpass=f=80,lowpass=f=8000,dynaudnorm=f=200:g=31,volume=20dB",
                        "-ac", "1",
                        "-ar", "16000",
                        boosted_wav,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                result = self._transcribe(boosted_wav, language=None)
                text = (result.get("text") or "").strip()

            if len(text) < 3:
                raise RuntimeError(
                    "No speech detected in recording. "
                    f"audio_stats(duration={duration:.2f}s,rms={rms:.6f},peak={peak:.6f}). "
                    "Please verify browser microphone input device and volume."
                )

            display_filename = source_filename or audio_path

            if format_type:
                metadata = self.legal_formatter._extract_metadata(text)
                formatter = self.legal_formatter.court_formats.get(
                    format_type, self.legal_formatter.format_high_court
                )
                import librosa
                duration = librosa.get_duration(path=normalized_wav)
                chunk_meta = [{"duration": duration}]
                formatted_doc = formatter(text, metadata, display_filename, chunk_meta)
                formatted = {
                    "document": formatted_doc,
                    "metadata": metadata,
                    "court_type": format_type,
                    "filename": display_filename,
                    "timestamp": datetime.now().isoformat(),
                    "total_chunks": len(chunk_meta),
                }
            else:
                import librosa
                duration = librosa.get_duration(path=normalized_wav)
                chunk_meta = [{"duration": duration}]
                formatted = self.legal_formatter.format_complete_document(
                    text,
                    chunks=chunk_meta,
                    filename=display_filename
                )

            # Keep duration available in API payload as well.
            duration = chunk_meta[0]["duration"] if chunk_meta else 0.0

            return {
                "text": text,
                "formatted": formatted,
                "language": result.get("language", language),
                "confidence": 1.0,
                "duration": duration,
                "audio_stats": {
                    "duration_seconds": round(duration, 3),
                    "rms": round(rms, 6),
                    "peak": round(peak, 6),
                },
            }
        finally:
            for path in (normalized_wav, boosted_wav):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass


_shared_transcriber = None
_shared_lock = threading.Lock()


def get_shared_transcriber(model_size: str = "base") -> JudicialTranscriber:
    """
    Return one process-wide transcriber instance.
    Remark: prevents loading Whisper model twice (WS + HTTP paths).
    """
    global _shared_transcriber
    if _shared_transcriber is None:
        with _shared_lock:
            if _shared_transcriber is None:
                _shared_transcriber = JudicialTranscriber(model_size=model_size)
    return _shared_transcriber
