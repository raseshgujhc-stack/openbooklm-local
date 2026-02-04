# rag/semantic_engine.py

from rag.llm import ask_llm, llm_generate_text
from rag.vector_store import load_vectors, get_collection_notebooks
from rag.embedder import embed_texts
import numpy as np

MAX_CHUNKS_PER_NOTEBOOK = 3
MAX_TEXT_CHARS = 6000

def handle_semantic_query(question, collection_id, user_id):
    notebook_ids = get_collection_notebooks(collection_id, user_id)
    if not notebook_ids:
        return "No documents found."

    q_emb = embed_texts([question])[0]
    q = np.array([q_emb], dtype="float32")

    partial_answers = []

    for nb_id in notebook_ids:
        loaded = load_vectors(nb_id)
        if not loaded:
            continue

        index, metadata = loaded
        if index.ntotal == 0:
            continue

        distances, indices = index.search(q, MAX_CHUNKS_PER_NOTEBOOK)

        chunks = []
        for idx in indices[0]:
            if idx != -1 and idx < len(metadata):
                chunks.append(metadata[idx]["text"])

        if not chunks:
            continue

        context = "\n\n".join(chunks)[:MAX_TEXT_CHARS]
        answer = ask_llm(context, question)

        if "not mentioned" not in answer.lower():
            partial_answers.append(answer)

    if not partial_answers:
        return "Not mentioned in the documents."

    synthesis_prompt = f"""
You are given extracted answers from multiple documents.

Combine them conservatively.
Do not invent facts.
If all answers align, state the conclusion.
If unclear, say so.

Extracted Answers:
------------------
{chr(10).join(partial_answers)}
------------------

Final Answer:
"""

    return llm_generate_text(synthesis_prompt).strip()

