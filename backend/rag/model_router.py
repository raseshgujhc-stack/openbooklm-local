import os
import threading
from pathlib import Path

from llama_cpp import Llama

_BASE_DIR = Path(__file__).resolve().parent.parent
_MODEL_DIR = _BASE_DIR / "models"

QWEN_MODEL_PATH = Path(
    os.getenv("QWEN_MODEL_PATH", str(_MODEL_DIR / "qwen2.5-7b-instruct-q3_k_m.gguf"))
)

LLM_THREADS = int(os.getenv("LLM_THREADS", str(max(1, (os.cpu_count() or 8) - 1))))
LLM_BATCH = int(os.getenv("LLM_N_BATCH", "256"))
LLM_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "0"))
QWEN_N_CTX = int(os.getenv("QWEN_N_CTX", "8192"))

_qwen = None
_lock = threading.Lock()


def _get_qwen() -> Llama:
    global _qwen
    if _qwen is None:
        with _lock:
            if _qwen is None:
                if not QWEN_MODEL_PATH.exists():
                    raise FileNotFoundError(f"Qwen model missing at: {QWEN_MODEL_PATH}")
                _qwen = Llama(
                    model_path=str(QWEN_MODEL_PATH),
                    n_ctx=QWEN_N_CTX,
                    n_threads=LLM_THREADS,
                    n_batch=LLM_BATCH,
                    n_gpu_layers=LLM_GPU_LAYERS,
                    verbose=False,
                )
    return _qwen


def qwen_summary(prompt: str, max_tokens: int = 220, temperature: float = 0.2) -> str:
    out = _get_qwen()(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        stop=["</s>", "<|endoftext|>"],
        echo=False,
    )
    return out["choices"][0]["text"].strip()


def qwen_podcast_script(prompt: str, max_tokens: int = 1400) -> str:
    out = _get_qwen()(
        prompt,
        max_tokens=max_tokens,
        temperature=0.35,
        top_p=0.9,
        stop=["</s>", "<|endoftext|>"],
        echo=False,
    )
    return out["choices"][0]["text"].strip()
