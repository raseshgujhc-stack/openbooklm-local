# rag/rag_pipeline.py

import numpy as np

from rag.llm import ask_llm, llm_generate_text
from rag.vector_store import load_vectors, get_collection_notebooks
from rag.embedder import embed_texts

from rag.question_router import classify_question
from rag.intent_classifier import classify_intent
from rag.metadata_engine import handle_metadata_intent


def generate_answer(
    question,
    relevant_chunks=None,
    notebook_id=None,
    collection_id=None,
    user_id=None,
):
    """
    Unified entry point:
    - Metadata queries → SQL (fast, deterministic)
    - Semantic queries → FAISS + LLM (existing logic)
    """

    # ==================================================
    # HARD QUESTION ROUTER (DETERMINISTIC)
    # ==================================================

    router_decision = classify_question(question)

    print("🔎 Router decision:", router_decision)

    if router_decision["route"] == "metadata":
        intent = {
            "intent_type": "metadata",
            "operation": router_decision.get("operation"),
            "entities": router_decision.get("entities", {}),
            "filters": {}
        }
    else:
        # Soft intent classification (LLM)
        intent = classify_intent(question)

    print("🧠 Final intent:", intent)

    # ==================================================
    # METADATA HANDLING (EARLY EXIT)
    # ==================================================

    if intent.get("intent_type") == "metadata":
        answer = handle_metadata_intent(
            intent=intent,
            collection_id=collection_id,
            user_id=user_id,
            notebook_id=notebook_id,
        )

        # If metadata engine handled it, stop here
        if answer and "not yet supported" not in answer.lower():
            return answer
        # else → fallback to semantic RAG

    # ==================================================
    # SEMANTIC RAG (UNCHANGED CORE LOGIC)
    # ==================================================

    MAX_CHUNKS_PER_NOTEBOOK = 3   # speed + stability
    MAX_TEXT_CHARS = 6000

    # ==================================================
    # COLLECTION MODE
    # ==================================================
    if collection_id and user_id:
        notebook_ids = get_collection_notebooks(collection_id, user_id)
        if not notebook_ids:
            return "Not mentioned in the documents."

        # Embed question ONCE
        q_emb = embed_texts([question])[0]
        q = np.array([q_emb], dtype="float32")

        partial_answers = []
        total = len(notebook_ids)

        for idx, nb_id in enumerate(notebook_ids, start=1):
            print(f"📘 Processing notebook {idx}/{total}: {nb_id}")

            loaded = load_vectors(nb_id)
            if not loaded:
                continue

            index, metadata = loaded
            if index.ntotal == 0:
                continue

            # ---- FAISS select best chunk(s) ----
            distances, indices = index.search(q, MAX_CHUNKS_PER_NOTEBOOK)

            chunks = []
            for meta_idx in indices[0]:
                if meta_idx == -1 or meta_idx >= len(metadata):
                    continue
                chunks.append(metadata[meta_idx]["text"])

            if not chunks:
                continue

            context = "\n\n".join(chunks)
            if len(context) > MAX_TEXT_CHARS:
                context = context[:MAX_TEXT_CHARS]

            # ✅ STRICT Q&A (unchanged)
            answer = ask_llm(context, question)

            if answer and "not mentioned" not in answer.lower():
                partial_answers.append(
                    f"Notebook {nb_id}:\n{answer}"
                )

        if not partial_answers:
            return "Not mentioned in the documents."

        # ==================================================
        # COLLECTION-LEVEL SYNTHESIS
        # ==================================================

        synthesis_prompt = f"""
You are given answers extracted independently from multiple documents.

Your task is to provide a collection-level understanding.

INSTRUCTIONS:
- Look for common patterns, agreements, or themes
- If most documents point to the same conclusion, state it clearly
- If documents differ or information is insufficient, say so explicitly
- Do NOT invent facts not present in the extracted answers
- Be concise and neutral

EXTRACTED ANSWERS:
----------------
{chr(10).join(partial_answers)}
----------------

ORIGINAL QUESTION:
{question}

Final synthesized answer (collection-level):
"""

        final_answer = llm_generate_text(synthesis_prompt)
        return final_answer.strip()

    # ==================================================
    # SINGLE NOTEBOOK MODE
    # ==================================================
    if notebook_id:
        from rag.similarity import similarity_search

        results = similarity_search(
            question=question,
            notebook_id=notebook_id,
            TOP_K=5,
        )
        if not results:
            return "Not mentioned in the document."

        context = "\n\n".join(r["text"] for r in results)
        if len(context) > MAX_TEXT_CHARS:
            context = context[:MAX_TEXT_CHARS]

        return ask_llm(context, question)

    return "No documents provided for answering."

