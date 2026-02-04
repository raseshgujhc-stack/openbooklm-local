import faiss
import json
import numpy as np
from rag.chunker import chunk_text
from rag.embedder import embed_texts
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sqlite3


BASE_DIR = Path(__file__).parent.parent / "data" / "faiss"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SAVE (BACKWARD COMPATIBLE)
# ============================================================

def save_vectors(
    notebook_id: str,
    vectors: List[Dict],
    collection_id: Optional[str] = None,
):
    """
    vectors = [
        {
            "text": "...",
            "embedding": [...],
            "chunk_index": int (optional)
        },
        ...
    ]

    ⚠️ notebook_id behavior preserved
    ✅ metadata added silently
    """

    embeddings = np.array(
        [v["embedding"] for v in vectors],
        dtype="float32",
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # --- Persist index ---
    faiss.write_index(
        index,
        str(BASE_DIR / f"{notebook_id}.index"),
    )

    # --- Persist metadata ---
    metadata = []
    for i, v in enumerate(vectors):
        metadata.append({
            "text": v["text"],
            "notebook_id": notebook_id,
            "collection_id": collection_id,   # future use
            "chunk_index": v.get("chunk_index", i),
        })

    with open(
        BASE_DIR / f"{notebook_id}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# ============================================================
# LOAD INDEX + METADATA (UNCHANGED API)
# ============================================================

def load_vectors(notebook_id: str):
    """
    Returns:
        index, metadata_list
    """
    index_path = BASE_DIR / f"{notebook_id}.index"
    meta_path = BASE_DIR / f"{notebook_id}.json"

    if not index_path.exists() or not meta_path.exists():
        return None

    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    return index, metadata


# ============================================================
# PODCAST SUPPORT (UNCHANGED)
# ============================================================

def load_texts(notebook_id: str) -> List[str]:
    """
    Load all text chunks for podcast generation.
    ⚠️ Existing behavior preserved
    """
    meta_path = BASE_DIR / f"{notebook_id}.json"

    if not meta_path.exists():
        return []

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return [m["text"] for m in metadata]


# ============================================================
# QUERY (NEW – SAFE TO ADD)
# ============================================================

def query_vectors(
    notebook_id: str,
    query_embedding: List[float],
    top_k: int = 10,
):
    """
    Semantic search inside ONE notebook.
    Collection-aware in future.
    """

    loaded = load_vectors(notebook_id)
    if not loaded:
        return []

    index, metadata = loaded

    q = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(q, top_k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(metadata[idx])

    return results


# ============================================================
# DELETE (UNCHANGED)
# ============================================================

def delete_vectors(notebook_id: str):
    index_path = BASE_DIR / f"{notebook_id}.index"
    meta_path = BASE_DIR / f"{notebook_id}.json"

    if index_path.exists():
        index_path.unlink()

    if meta_path.exists():
        meta_path.unlink()
#=============================================================
# Build Vectors
#============================================================

def build_vectors(notebook_id, full_text):
    chunks = chunk_text(full_text)

    embeddings = embed_texts(chunks)

    vectors = []
    for text, emb in zip(chunks, embeddings):
        vectors.append({
            "text": text,
            "embedding": emb,
        })

    return vectors


# ============================================================
# COLLECTION-AWARE FUNCTIONS
# ============================================================

def get_collection_notebooks(collection_id: str, user_id: str) -> List[str]:
    """Get all notebook IDs in a collection"""
    conn = sqlite3.connect('data/notebooks.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT notebook_id FROM notebooks 
        WHERE collection_id = ? AND user_id = ?
    """, (collection_id, user_id))
    
    notebook_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return notebook_ids

def search_across_collection(
    collection_id: str,
    query_embedding: List[float],
    top_k: int = 10,
    user_id: str = None
) -> List[Dict]:
    """
    Search across all notebooks in a collection.
    GUARANTEES at least ONE result per notebook.
    Designed for document-wise extraction.
    """

    notebook_ids = get_collection_notebooks(collection_id, user_id)
    if not notebook_ids:
        return []

    all_results = []
    q = np.array([query_embedding], dtype="float32")

    for notebook_id in notebook_ids:
        loaded = load_vectors(notebook_id)
        if not loaded:
            continue

        index, metadata = loaded
        if index.ntotal == 0:
            continue

        distances, indices = index.search(q, 1)

        idx = indices[0][0]
        if idx == -1 or idx >= len(metadata):
            continue

        distance = float(distances[0][0])

        result = metadata[idx].copy()
        result["notebook_id"] = notebook_id
        result["distance"] = distance
        result["score"] = 1.0 / (1.0 + distance)

        all_results.append(result)

    return all_results



def save_to_collection(
    notebook_id: str,
    vectors: List[Dict],
    collection_id: str,
    filename: str = None
):
    """
    Save vectors with collection metadata
    """
    # Save to FAISS (existing function)
    save_vectors(notebook_id, vectors, collection_id)
    
    # Update database with collection info
    conn = sqlite3.connect('data/notebooks.db')
    cursor = conn.cursor()
    
    if filename:
        cursor.execute("""
            UPDATE notebooks 
            SET collection_id = ?, filename = ?
            WHERE notebook_id = ?
        """, (collection_id, filename, notebook_id))
    else:
        cursor.execute("""
            UPDATE notebooks 
            SET collection_id = ?
            WHERE notebook_id = ?
        """, (collection_id, notebook_id))
    
    conn.commit()
    conn.close()

# ============================================================
# ENHANCED LOAD FUNCTION
# ============================================================

def load_collection_vectors(collection_id: str, user_id: str = None):
    """
    Load all vectors from all notebooks in a collection
    Returns combined metadata list
    """
    notebook_ids = get_collection_notebooks(collection_id, user_id)
    
    all_metadata = []
    
    for notebook_id in notebook_ids:
        meta_path = BASE_DIR / f"{notebook_id}.json"
        
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            
            # Add notebook_id to each metadata entry
            for meta in metadata:
                meta["notebook_id"] = notebook_id
                all_metadata.append(meta)
    
    return all_metadata
