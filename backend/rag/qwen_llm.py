import os

from llama_cpp import Llama

MODEL_PATH = "./models/qwen2.5-7b-instruct-q3_k_m.gguf"
QWEN_RAG_ANSWER_MAX_TOKENS = int(os.getenv("QWEN_RAG_ANSWER_MAX_TOKENS", "700"))

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=int(os.getenv("QWEN_N_CTX", "8192")),
    n_threads=int(os.getenv("QWEN_THREADS", str(max(1, (os.cpu_count() or 8) - 1)))),
    n_batch=int(os.getenv("QWEN_N_BATCH", "512")),
    verbose=False,
)

# ===============================
# STRICT DOCUMENT Q&A (RAG ONLY)
# ===============================
def ask_llm(context: str, question: str, max_tokens: int | None = None) -> str:
    prompt = f"""You are a judicial information extraction system.

RULES:
- Use ONLY the provided document text
- Do NOT guess
- If the answer is not explicitly present, reply exactly:
  "Not mentioned in the document."

Document Text:
----------------
{context}
----------------

Question:
{question}

Answer:"""

    response = llm(
        prompt,
        temperature=0.0,
        # Remark: avoid truncating long legal answers by default.
        max_tokens=max_tokens or QWEN_RAG_ANSWER_MAX_TOKENS,
        stop=[
            "Answer:",
            "Document Text:",
            "<|endoftext|>",
            "</s>",
        ],
    )

    return response["choices"][0]["text"].strip()


# ===============================
# FREE GENERATION
# ===============================
def llm_generate_text(prompt: str) -> str:
    full_prompt = f"""You are a helpful legal assistant.

{prompt}
"""

    response = llm(
        full_prompt,
        temperature=0.3,
        max_tokens=800,
        stop=[
            "<|endoftext|>",
            "</s>",
        ],
    )

    return response["choices"][0]["text"].strip()
