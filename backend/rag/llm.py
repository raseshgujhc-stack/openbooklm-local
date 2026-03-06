import os
from llama_cpp import Llama

MODEL_PATH = "./models/mistral-7b-instruct.gguf"
LLM_N_CTX = int(os.getenv("MISTRAL_N_CTX", "8192"))
LLM_N_THREADS = int(os.getenv("MISTRAL_THREADS", str(max(1, (os.cpu_count() or 8) - 1))))
RAG_ANSWER_MAX_TOKENS = int(os.getenv("RAG_ANSWER_MAX_TOKENS", "700"))

llm = Llama(
    model_path=MODEL_PATH,
    # Remark: n_ctx is runtime window; keep configurable for CPU/RAM tradeoff.
    n_ctx=LLM_N_CTX,
    n_threads=LLM_N_THREADS,
    verbose=False,
)

# ===============================
# STRICT DOCUMENT Q&A (RAG ONLY)
# ===============================
def ask_llm(context: str, question: str, max_tokens: int | None = None) -> str:
    prompt = f"""
You are an information extraction system.

CRITICAL RULES:
- Use ONLY the provided document text
- DO NOT use prior knowledge
- DO NOT guess
- DO NOT infer
- If the answer is not explicitly present, reply exactly:
  "Not mentioned in the document."

Document Text:
----------------
{context}
----------------

Question:
{question}

Answer:
"""

    output = llm(
        prompt,
        # Remark: keep output budget configurable to avoid mid-answer truncation.
        max_tokens=max_tokens or RAG_ANSWER_MAX_TOKENS,
        temperature=0.0,      # 🚨 VERY IMPORTANT
        top_p=1.0,
        stop=["Question:", "Document Text:"],
        echo=False,
    )

    return output["choices"][0]["text"].strip()


# ===============================
# FREE GENERATION (PODCAST, ETC.)
# ===============================
def llm_generate_text(prompt: str) -> str:
    response = llm(
        prompt,
        max_tokens=1200,
        temperature=0.7,
        top_p=0.9,
        stop=["</s>"],
        echo=False,
    )

    return response["choices"][0]["text"].strip()
