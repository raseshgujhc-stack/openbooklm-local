# rag/llm.py

from llama_cpp import Llama

MODEL_PATH = "./models/qwen2.5-7b-instruct-q3_k_m.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_threads=16,
    n_batch=512,
    verbose=False,
)

# ===============================
# STRICT DOCUMENT Q&A (RAG ONLY)
# ===============================
def ask_llm(context: str, question: str) -> str:
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
        max_tokens=256,
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

