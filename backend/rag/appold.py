from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    Header,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import bcrypt
import threading
import json
from pathlib import Path
import os
from fastapi.responses import StreamingResponse

from rag.podcast import generate_podcast_script
from rag.db import get_db, init_db, init_chat_table
from rag.pdf_reader import read_pdf
from rag.text_splitter import split_text
from rag.embedder import embed
from rag.rag_pipeline import generate_answer
from rag.vector_store import save_vectors, load_vectors, delete_vectors, load_texts, get_collection_notebooks
from rag.podcast_worker import run_podcast_job
from rag.auth import get_current_user_id
from typing import Optional
from rag.ingest import ingest_document


BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# App init
# --------------------------------------------------

app = FastAPI(title="Local NotebookLM Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
init_chat_table()

# --------------------------------------------------
# Models
# --------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str

class CreateCollectionRequest(BaseModel):
    name: str

class PodcastRequest(BaseModel):
    notebook_id: str
    speakers: int = 2

class CollectionChatRequest(BaseModel):
    collection_id: str
    question: str

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def verify_password(plain: str, hashed: str) -> bool:
    # bcrypt max = 72 bytes
    return bcrypt.checkpw(
        plain.encode("utf-8")[:72],
        hashed.encode("utf-8"),
    )

def get_current_user_id(x_user_id: str = Header(...)):
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing X-User-Id")
    return x_user_id

# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok"}

# --------------------------------------------------
# Login
# --------------------------------------------------

@app.post("/login")
def login(data: LoginRequest):
    db = get_db()
    user = db.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (data.username,),
    ).fetchone()
    db.close()

    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "user_id": user["id"],
        "username": user["username"],
    }

# Upload PDF
# --------------------------------------------------
@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    collection_id: Optional[str] = Form(None),  # ✅ optional
    user_id: str = Depends(get_current_user_id),
):
    text = read_pdf(file)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty PDF")

    # 🔐 COLLECTION OWNERSHIP CHECK (ONLY IF PROVIDED)
    if collection_id:
        db = get_db()
        owned = db.execute(
            """
            SELECT 1 FROM collections
            WHERE collection_id = ? AND user_id = ?
            """,
            (collection_id, user_id),
        ).fetchone()

        if not owned:
            db.close()
            raise HTTPException(
                status_code=403,
                detail="Collection not found or not owned by user",
            )

        db.close()

    # 🔹 Normal PDF pipeline (semantic + metadata)
    chunks = split_text(text)
    notebook_id = str(uuid.uuid4())

    # ---- Semantic ingestion (UNCHANGED) ----
    embeddings = embed(chunks)
    vectors = [{"text": c, "embedding": e} for c, e in zip(chunks, embeddings)]
    save_vectors(notebook_id, vectors)

    # ---- Metadata ingestion (NEW – REQUIRED) ----
    ingest_document(
        text=text,                     # full document text
        document_id=notebook_id,        # IMPORTANT: keep IDs aligned
        collection_id=collection_id,
        user_id=user_id,
        filename=file.filename,
    )

    # ---- Notebooks table (UNCHANGED) ----
    db = get_db()
    db.execute(
        """
        INSERT INTO notebooks (notebook_id, filename, user_id, collection_id)
        VALUES (?, ?, ?, ?)
        """,
        (notebook_id, file.filename, user_id, collection_id),
    )
    db.commit()
    db.close()

# --------------------------------------------------
# List notebooks
# --------------------------------------------------

@app.get("/notebooks")
def list_notebooks(user_id: str = Depends(get_current_user_id)):
    db = get_db()
    rows = db.execute(
        """
        SELECT notebook_id, filename, created_at, collection_id
        FROM notebooks
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,),
    ).fetchall()
    db.close()

    notebooks = []
    for r in rows:
        notebook = dict(r)
        # Get collection name if exists
        if notebook["collection_id"]:
            db2 = get_db()
            collection = db2.execute(
                "SELECT name FROM collections WHERE collection_id = ?",
                (notebook["collection_id"],)
            ).fetchone()
            db2.close()
            notebook["collection_name"] = collection["name"] if collection else None
        else:
            notebook["collection_name"] = None
        notebooks.append(notebook)
    
    return notebooks

# --------------------------------------------------
# Single PDF Chat
# --------------------------------------------------

@app.post("/pdf-chat")
def pdf_chat(
    payload: dict,
    user_id: str = Depends(get_current_user_id),
):
    notebook_id = payload.get("notebook_id")
    question = payload.get("question")

    if not notebook_id or not question:
        raise HTTPException(status_code=400, detail="Missing data")

    db = get_db()
    owned = db.execute(
        "SELECT 1 FROM notebooks WHERE notebook_id=? AND user_id=?",
        (notebook_id, user_id),
    ).fetchone()

    if not owned:
        db.close()
        raise HTTPException(status_code=403, detail="Forbidden")

    loaded = load_vectors(notebook_id)
    if loaded:
        index, metadata = loaded
        vectors = metadata
    else:
        vectors = []

    answer = generate_answer(question, vectors, notebook_id)

    db.execute(
        "INSERT INTO chat_history (notebook_id, role, content) VALUES (?, ?, ?)",
        (notebook_id, "user", question),
    )
    db.execute(
        "INSERT INTO chat_history (notebook_id, role, content) VALUES (?, ?, ?)",
        (notebook_id, "assistant", answer),
    )
    db.commit()
    db.close()

    return {"answer": answer}

# --------------------------------------------------
# Collection Chat (NEW)
# --------------------------------------------------

@app.post("/collection-chat")
def collection_chat(
    payload: CollectionChatRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Chat with an entire collection of documents
    """
    if not payload.collection_id or not payload.question:
        raise HTTPException(status_code=400, detail="Missing collection_id or question")
    
    # Verify collection ownership
    db = get_db()
    owned = db.execute(
        "SELECT name FROM collections WHERE collection_id=? AND user_id=?",
        (payload.collection_id, user_id),
    ).fetchone()
    
    if not owned:
        db.close()
        raise HTTPException(status_code=403, detail="Collection not found or not owned")
    
    collection_name = owned["name"]
    db.close()
    
    # Generate answer using collection-aware pipeline
    answer = generate_answer(
        question=payload.question,
        collection_id=payload.collection_id,
        user_id=user_id
    )
    
    # Store in chat history (create table if not exists)
    db = get_db()
    try:
        db.execute(
            "INSERT INTO collection_chat_history (collection_id, user_id, role, content) VALUES (?, ?, ?, ?)",
            (payload.collection_id, user_id, "user", payload.question),
        )
        db.execute(
            "INSERT INTO collection_chat_history (collection_id, user_id, role, content) VALUES (?, ?, ?, ?)",
            (payload.collection_id, user_id, "assistant", answer),
        )
        db.commit()
    except Exception as e:
        # Table might not exist, but that's OK for now
        print(f"Note: collection_chat_history table not available: {e}")
        pass
    finally:
        db.close()
    
    # Get notebook count for info
    notebook_ids = get_collection_notebooks(payload.collection_id, user_id)
    
    return {
        "answer": answer,
        "collection_id": payload.collection_id,
        "collection_name": collection_name,
        "sources": len(notebook_ids)
    }

# --------------------------------------------------
# Delete notebook
# --------------------------------------------------

@app.delete("/notebook/{notebook_id}")
def delete_notebook(
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    db = get_db()
    owned = db.execute(
        "SELECT 1 FROM notebooks WHERE notebook_id=? AND user_id=?",
        (notebook_id, user_id),
    ).fetchone()

    if not owned:
        db.close()
        raise HTTPException(status_code=404, detail="Not found")

    db.execute("DELETE FROM notebooks WHERE notebook_id=?", (notebook_id,))
    db.execute("DELETE FROM chat_history WHERE notebook_id=?", (notebook_id,))
    db.commit()
    db.close()

    delete_vectors(notebook_id)
    return {"status": "deleted"}

# --------------------------------------------------
# Chat history
# --------------------------------------------------

@app.get("/chat-history/{notebook_id}")
def get_chat_history(
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    db = get_db()
    rows = db.execute(
        """
        SELECT role, content, created_at
        FROM chat_history
        WHERE notebook_id = ?
        AND notebook_id IN (
            SELECT notebook_id FROM notebooks WHERE user_id = ?
        )
        ORDER BY created_at ASC
        """,
        (notebook_id, user_id),
    ).fetchall()
    db.close()

    return [dict(r) for r in rows]

# --------------------------------------------------
# Collection Contents (NEW)
# --------------------------------------------------

@app.get("/collection/{collection_id}/contents")
def get_collection_contents(
    collection_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Get all notebooks and their info in a collection
    """
    db = get_db()
    
    # Verify ownership
    owned = db.execute(
        "SELECT name FROM collections WHERE collection_id=? AND user_id=?",
        (collection_id, user_id),
    ).fetchone()
    
    if not owned:
        db.close()
        raise HTTPException(status_code=403, detail="Collection not found or not owned")
    
    collection_name = owned["name"]
    
    # Get all notebooks in collection
    notebooks = db.execute(
        """
        SELECT notebook_id, filename, created_at
        FROM notebooks
        WHERE collection_id=? AND user_id=?
        ORDER BY created_at
        """,
        (collection_id, user_id),
    ).fetchall()
    
    db.close()
    
    # Get chunk counts for each notebook
    notebook_details = []
    for nb in notebooks:
        meta_path = BASE_DIR / "data" / "faiss" / f"{nb['notebook_id']}.json"
        chunk_count = 0
        
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                chunk_count = len(metadata)
            except:
                chunk_count = 0
        
        notebook_details.append({
            "notebook_id": nb["notebook_id"],
            "filename": nb["filename"],
            "created_at": nb["created_at"],
            "chunk_count": chunk_count
        })
    
    return {
        "collection_id": collection_id,
        "collection_name": collection_name,
        "notebooks": notebook_details,
        "total_notebooks": len(notebook_details)
    }

# --------------------------------------------------
# Search Across Collection (NEW)
# --------------------------------------------------

@app.post("/collection/{collection_id}/search")
def search_collection(
    collection_id: str,
    payload: dict,
    user_id: str = Depends(get_current_user_id),
):
    """
    Semantic search across collection (returns raw results, not answers)
    """
    from rag.similarity import similarity_search
    
    question = payload.get("question", "")
    top_k = payload.get("top_k", 10)
    
    if not question:
        raise HTTPException(status_code=400, detail="Missing question")
    
    # Verify ownership
    db = get_db()
    owned = db.execute(
        "SELECT name FROM collections WHERE collection_id=? AND user_id=?",
        (collection_id, user_id),
    ).fetchone()
    
    if not owned:
        db.close()
        raise HTTPException(status_code=403, detail="Collection not found or not owned")
    
    collection_name = owned["name"]
    db.close()
    
    # Perform search
    results = similarity_search(
        question=question,
        collection_id=collection_id,
        user_id=user_id,
        TOP_K=top_k
    )
    
    # Format results with preview
    formatted_results = []
    for result in results:
        text_preview = result.get("text", "")
        if len(text_preview) > 300:
            text_preview = text_preview[:300] + "..."
        
        formatted_results.append({
            "text": result.get("text", ""),
            "preview": text_preview,
            "score": round(result.get("score", 0), 4),
            "notebook_id": result.get("notebook_id", ""),
            "source": result.get("source", "Unknown"),
            "chunk_index": result.get("chunk_index", 0)
        })
    
    return {
        "collection_id": collection_id,
        "collection_name": collection_name,
        "question": question,
        "total_results": len(results),
        "results": formatted_results
    }

# --------------------------------------------------
# Collection History (NEW)
# --------------------------------------------------

@app.get("/collection/{collection_id}/history")
def get_collection_chat_history(
    collection_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Get chat history for a collection
    """
    db = get_db()
    
    # Verify ownership
    owned = db.execute(
        "SELECT name FROM collections WHERE collection_id=? AND user_id=?",
        (collection_id, user_id),
    ).fetchone()
    
    if not owned:
        db.close()
        raise HTTPException(status_code=403, detail="Collection not found or not owned")
    
    # Try to get from collection_chat_history table
    try:
        rows = db.execute(
            """
            SELECT role, content, created_at
            FROM collection_chat_history
            WHERE collection_id = ? AND user_id = ?
            ORDER BY created_at ASC
            """,
            (collection_id, user_id),
        ).fetchall()
        
        if rows:
            history = [dict(r) for r in rows]
        else:
            # Fallback to empty list if table doesn't exist or no history
            history = []
            
    except Exception:
        # Table might not exist
        history = []
    
    db.close()
    
    return {
        "collection_id": collection_id,
        "collection_name": owned["name"],
        "history": history
    }

# --------------------------------------------------
# Create Collection
# --------------------------------------------------

@app.post("/collections")
def create_collection(
    data: CreateCollectionRequest,
    user_id: str = Depends(get_current_user_id),
):
    collection_id = str(uuid.uuid4())
    db = get_db()

    db.execute(
        """
        INSERT INTO collections (collection_id, name, user_id)
        VALUES (?, ?, ?)
        """,
        (collection_id, data.name, user_id),
    )
    db.commit()
    db.close()

    return {
        "collection_id": collection_id,
        "name": data.name,
    }

# --------------------------------------------------
# List Collections
# --------------------------------------------------

@app.get("/collections")
def list_collections(
    user_id: str = Depends(get_current_user_id),
):
    db = get_db()
    rows = db.execute(
        """
        SELECT 
            c.collection_id, 
            c.name, 
            c.created_at,
            COUNT(n.notebook_id) as notebook_count
        FROM collections c
        LEFT JOIN notebooks n ON c.collection_id = n.collection_id AND n.user_id = c.user_id
        WHERE c.user_id=?
        GROUP BY c.collection_id
        ORDER BY c.created_at DESC
        """,
        (user_id,),
    ).fetchall()
    db.close()

    return [
        {
            "collection_id": r["collection_id"],
            "name": r["name"],
            "created_at": r["created_at"],
            "notebook_count": r["notebook_count"] or 0
        }
        for r in rows
    ]

# --------------------------------------------------
# Delete Collection
# --------------------------------------------------

@app.delete("/collections/{collection_id}")
def delete_collection(
    collection_id: str,
    user_id: str = Depends(get_current_user_id),
):
    db = get_db()

    owned = db.execute(
        """
        SELECT 1 FROM collections
        WHERE collection_id=? AND user_id=?
        """,
        (collection_id, user_id),
    ).fetchone()

    if not owned:
        db.close()
        raise HTTPException(status_code=404, detail="Collection not found")

    # unlink notebooks
    db.execute(
        """
        UPDATE notebooks
        SET collection_id=NULL
        WHERE collection_id=? AND user_id=?
        """,
        (collection_id, user_id),
    )

    # delete collection
    db.execute(
        "DELETE FROM collections WHERE collection_id=?",
        (collection_id,),
    )
    
    # Clean up collection chat history
    try:
        db.execute(
            "DELETE FROM collection_chat_history WHERE collection_id=?",
            (collection_id,),
        )
    except:
        pass  # Table might not exist

    db.commit()
    db.close()

    return {"status": "deleted"}

# --------------------------------------------------
# PODCAST ENDPOINTS (Existing - Unchanged)
# --------------------------------------------------

@app.post("/podcast/generate")
def generate_podcast(
    data: PodcastRequest,
    user_id: str = Depends(get_current_user_id),
):
    db = get_db()

    owned = db.execute(
        "SELECT 1 FROM notebooks WHERE notebook_id=? AND user_id=?",
        (data.notebook_id, user_id),
    ).fetchone()

    if not owned:
        db.close()
        raise HTTPException(status_code=403, detail="Forbidden")

    job_id = str(uuid.uuid4())

    db.execute(
        """
        INSERT INTO podcast_jobs (id, notebook_id, user_id, status, speakers)
        VALUES (?, ?, ?, ?, ?)
        """,
        (job_id, data.notebook_id, user_id, "pending", data.speakers),
    )
    db.commit()
    db.close()

    threading.Thread(
        target=run_podcast_job,
        args=(job_id, data.notebook_id, user_id, data.speakers),
        daemon=True,
    ).start()

    return {"job_id": job_id}

@app.get("/podcast/status/{job_id}")
def podcast_status(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
):
    db = get_db()

    row = db.execute(
        """
        SELECT status, result
        FROM podcast_jobs
        WHERE id=? AND user_id=?
        """,
        (job_id, user_id),
    ).fetchone()

    db.close()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "status": row["status"],
        "result": row["result"] if row["status"] in ("script_ready", "done") else None,
    }

@app.get("/podcast/latest/{notebook_id}")
def get_latest_podcast(
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    db = get_db()

    row = db.execute(
        """
        SELECT id, result, speakers
        FROM podcast_jobs
        WHERE notebook_id=?
          AND user_id=?
          AND status IN ('script_ready', 'done')
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (notebook_id, user_id),
    ).fetchone()

    db.close()

    if not row:
        raise HTTPException(status_code=404, detail="No podcast found")

    return {
        "job_id": row["id"],
        "result": row["result"],
        "speakers": row["speakers"],
    }

@app.get("/podcast/audio/{job_id}")
def get_podcast_audio(job_id: str):
    db = get_db()

    row = db.execute(
        "SELECT audio_path FROM podcast_jobs WHERE id=?",
        (job_id,),
    ).fetchone()

    db.close()

    if not row or not row["audio_path"]:
        raise HTTPException(status_code=404, detail="Audio not ready")

    audio_path = Path(row["audio_path"])

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing")

    file_size = os.path.getsize(audio_path)

    def iterfile():
        with open(audio_path, "rb") as f:
            yield from f

    return StreamingResponse(
        iterfile(),
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Disposition": f'inline; filename="{audio_path.name}"',
        },
    )
#-------- REMOVE NOTEBOOK FROM COLLECTION ENDPOINT ------------------------

@app.delete("/collection/{collection_id}/notebook/{notebook_id}")
def remove_notebook_from_collection(
    collection_id: str,
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Remove a notebook from a collection
    """
    db = get_db()
    
    # Verify ownership
    owned = db.execute(
        "SELECT 1 FROM collections WHERE collection_id=? AND user_id=?",
        (collection_id, user_id),
    ).fetchone()
    
    if not owned:
        db.close()
        raise HTTPException(status_code=403, detail="Collection not found or not owned")
    
    # Verify notebook exists and belongs to user
    notebook = db.execute(
        "SELECT 1 FROM notebooks WHERE notebook_id=? AND user_id=?",
        (notebook_id, user_id),
    ).fetchone()
    
    if not notebook:
        db.close()
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    # Remove from collection (set collection_id to NULL)
    db.execute(
        "UPDATE notebooks SET collection_id = NULL WHERE notebook_id = ? AND collection_id = ?",
        (notebook_id, collection_id),
    )
    db.commit()
    db.close()
    
    return {"status": "removed", "notebook_id": notebook_id, "collection_id": collection_id}
