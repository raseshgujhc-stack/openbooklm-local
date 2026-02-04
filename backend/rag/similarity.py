# file: similarity.py
import numpy as np
from rag.embedder import embed_query
from rag.vector_store import load_vectors, search_across_collection

def cosine_similarity(a, b):
    """Calculate cosine similarity"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

TOP_K = 5

def similarity_search(
    question, 
    vectors=None, 
    notebook_id=None, 
    collection_id=None,
    user_id=None,
    TOP_K=5
):
    """
    Enhanced similarity search that supports:
    1. Single notebook search
    2. Collection-wide search
    3. In-memory vector search
    """
    
    # Case 1: Collection search
    if collection_id and user_id:
        print(f"🔍 Searching across collection: {collection_id}")
        results = search_across_collection(
            collection_id=collection_id,
            query_embedding=embed_query(question),
            top_k=TOP_K,
            user_id=user_id
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result.get("text", ""),
                "score": result.get("score", 0.0),
                "notebook_id": result.get("notebook_id", ""),
                "chunk_index": result.get("chunk_index", 0),
                "source": f"Notebook: {result.get('notebook_id', '')[:8]}..."
            })
        
        return formatted_results
    
    # Case 2: Single notebook search (FAISS)
    elif notebook_id:
        loaded = load_vectors(notebook_id)
        if not loaded:
            return []

        index, metadata = loaded
        question_embedding = np.array(
            embed_query(question), dtype="float32"
        ).reshape(1, -1)

        distances, indices = index.search(question_embedding, TOP_K)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if idx < len(metadata):
                results.append({
                    "text": metadata[idx].get("text", ""),
                    "score": float(1 - dist),  # Convert distance to similarity
                    "notebook_id": notebook_id,
                    "chunk_index": metadata[idx].get("chunk_index", idx),
                    "source": f"Notebook: {notebook_id[:8]}..."
                })
        
        return results
    
    # Case 3: In-memory vectors
    elif vectors and isinstance(vectors[0], dict) and "embedding" in vectors[0]:
        question_embedding = embed_query(question)

        scored = []
        for v in vectors:
            if "embedding" not in v:
                continue
            score = cosine_similarity(question_embedding, v["embedding"])
            scored.append({
                "text": v["text"],
                "score": float(score),
                "source": "In-memory"
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:TOP_K]
    
    else:
        return []
