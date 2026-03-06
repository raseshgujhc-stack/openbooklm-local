import faiss
import json
import numpy as np
from rag.chunker import chunk_text
from rag.embedder import embed_texts
from pathlib import Path
from typing import List, Dict, Optional

from db import get_repo   # ✅ NEW


BASE_DIR = Path(__file__).parent.parent / "data" / "faiss"
BASE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SAVE (UNCHANGED)
# ============================================================

def save_vectors(
    notebook_id: str,
    vectors: List[Dict],
    collection_id: Optional[str] = None,
):
    embeddings = np.array(
        [v["embedding"] for v in vectors],
        dtype="float32",
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(
        index,
        str(BASE_DIR / f"{notebook_id}.index"),
    )

    metadata = []
    for i, v in enumerate(vectors):
        metadata.append({
            "text": v["text"],
            "notebook_id": notebook_id,
            "collection_id": collection_id,
            "chunk_index": v.get("chunk_index", i),
        })

    with open(
        BASE_DIR / f"{notebook_id}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# ============================================================
# LOAD INDEX + METADATA (UNCHANGED)
# ============================================================

def load_vectors(notebook_id: str):
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
    meta_path = BASE_DIR / f"{notebook_id}.json"

    if not meta_path.exists():
        return []

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return [m["text"] for m in metadata]


# ============================================================
# QUERY (UNCHANGED)
# ============================================================

def query_vectors(
    notebook_id: str,
    query_embedding: List[float],
    top_k: int = 10,
):
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


# ============================================================
# BUILD VECTORS (UNCHANGED)
# ============================================================

def build_vectors(notebook_id, full_text):
    chunks = chunk_text(full_text)
    embeddings = embed_texts(chunks)

    return [
        {"text": text, "embedding": emb}
        for text, emb in zip(chunks, embeddings)
    ]


# ============================================================
# COLLECTION-AWARE FUNCTIONS (POSTGRES)
# ============================================================
def get_collection_notebooks(collection_id: str, user_id: str) -> List[str]:
    repo = get_repo()
    conn = repo.conn
    cur = conn.cursor()

    cur.execute("""
        SELECT n.notebook_id
        FROM notebooks n
        JOIN collections c
          ON n.collection_id = c.collection_id
        WHERE n.collection_id = %s
          AND (
                n.user_id = %s
             OR c.is_global = TRUE
          )
    """, (collection_id, user_id))

    return [row[0] for row in cur.fetchall()]


def search_across_collection(
    collection_id: str,
    query_embedding: List[float],
    top_k: int = 10,
    user_id: str = None
) -> List[Dict]:
    """
    Search across all notebooks in a collection.
    GUARANTEES at least ONE result per notebook.
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
    Save vectors and update notebook collection (PostgreSQL)
    """
    save_vectors(notebook_id, vectors, collection_id)

    repo = get_repo()
    conn = repo.conn
    cur = conn.cursor()

    if filename:
        cur.execute("""
            UPDATE notebooks
            SET collection_id = %s, filename = %s
            WHERE notebook_id = %s
        """, (collection_id, filename, notebook_id))
    else:
        cur.execute("""
            UPDATE notebooks
            SET collection_id = %s
            WHERE notebook_id = %s
        """, (collection_id, notebook_id))

    conn.commit()


# ============================================================
# ENHANCED LOAD FUNCTION (UNCHANGED)
# ============================================================

def load_collection_vectors(collection_id: str, user_id: str = None):
    notebook_ids = get_collection_notebooks(collection_id, user_id)
    all_metadata = []

    for notebook_id in notebook_ids:
        meta_path = BASE_DIR / f"{notebook_id}.json"

        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

            for meta in metadata:
                meta["notebook_id"] = notebook_id
                all_metadata.append(meta)

    return all_metadata
    
    
def get_global_notebooks() -> List[str]:
    """
    Get all notebook IDs that belong to global collections.
    """
    repo = get_repo()
    conn = repo.conn
    cur = conn.cursor()

    cur.execute("""
        SELECT n.notebook_id
        FROM notebooks n
        JOIN collections c
          ON n.collection_id = c.collection_id
        WHERE c.is_global = TRUE
    """)

    return [row[0] for row in cur.fetchall()]

def build_global_index():
    """
    Build a single FAISS index for ALL global notebooks.
    """

    notebook_ids = get_global_notebooks()

    all_embeddings = []
    all_metadata = []

    for nb_id in notebook_ids:
        loaded = load_vectors(nb_id)
        if not loaded:
            continue

        index, metadata = loaded

        for i in range(index.ntotal):
            vector = index.reconstruct(i)
            all_embeddings.append(vector)
            all_metadata.append(metadata[i])

    if not all_embeddings:
        return

    embeddings = np.array(all_embeddings, dtype="float32")

    dim = embeddings.shape[1]
    global_index = faiss.IndexFlatL2(dim)
    global_index.add(embeddings)

    faiss.write_index(
        global_index,
        str(BASE_DIR / "global_master.index")
    )

    with open(BASE_DIR / "global_master.json", "w", encoding="utf-8") as f:
        json.dump(all_metadata, f)
def load_global_index():
    index_path = BASE_DIR / "global_master.index"
    meta_path = BASE_DIR / "global_master.json"

    if not index_path.exists():
        return None

    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    return index, metadata

