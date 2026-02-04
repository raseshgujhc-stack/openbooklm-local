from llama_cpp import Llama

MODEL_PATH = "./models/mistral-7b-instruct.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,
)

# ===============================
# STRICT DOCUMENT Q&A (RAG ONLY)
# ===============================
def ask_llm(context: str, question: str) -> str:
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
        max_tokens=256,
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
