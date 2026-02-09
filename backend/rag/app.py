from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    Header,
)
import threading
from ingest_worker import ingest_worker_loop
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import bcrypt
import threading
import json
from pathlib import Path
import os
from fastapi.responses import StreamingResponse
from typing import Optional

from db import get_repo   # ✅ PostgreSQL repo

from rag.podcast import generate_podcast_script
from rag.pdf_reader import read_pdf
from rag.text_splitter import split_text
from rag.embedder import embed
from rag.rag_pipeline import generate_answer
from rag.vector_store import (
    save_vectors,
    load_vectors,
    delete_vectors,
    load_texts,
    get_collection_notebooks,
)
from rag.podcast_worker import run_podcast_job
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
@app.on_event("startup")
def start_ingest_worker():
    thread = threading.Thread(
        target=ingest_worker_loop,
        daemon=True,
    )
    thread.start()

# ❌ init_db()
# ❌ init_chat_table()
# PostgreSQL schema already exists

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
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        SELECT id, username, password_hash
        FROM users
        WHERE username = %s
        """,
        (data.username,),
    )

    user = cur.fetchone()

    if not user or not verify_password(data.password, user[2]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "user_id": user[0],
        "username": user[1],
    }
# --------------------------------------------------
# Upload PDF
# --------------------------------------------------
@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    collection_id: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user_id),
):
    notebook_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    upload_dir = BASE_DIR / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{notebook_id}.pdf"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        INSERT INTO notebooks
        (notebook_id, filename, user_id, collection_id, status)
        VALUES (%s, %s, %s, %s, 'queued')
        """,
        (notebook_id, file.filename, user_id, collection_id),
    )

    cur.execute(
        """
        INSERT INTO ingest_jobs
        (job_id, notebook_id, user_id, status)
        VALUES (%s, %s, %s, 'queued')
        """,
        (job_id, notebook_id, user_id),
    )

    repo.conn.commit()

    return {
        "notebook_id": notebook_id,
        "status": "queued"
    }


# --------------------------------------------------
# List notebooks
# --------------------------------------------------

@app.get("/notebooks")
def list_notebooks(user_id: str = Depends(get_current_user_id)):
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        SELECT notebook_id, filename, created_at, collection_id
        FROM notebooks
        WHERE user_id = %s
        ORDER BY created_at DESC
        """,
        (user_id,),
    )
    rows = cur.fetchall()

    notebooks = []

    for r in rows:
        notebook = {
            "notebook_id": r[0],
            "filename": r[1],
            "created_at": r[2],
            "collection_id": r[3],
        }

        # Get collection name if exists
        if r[3]:
            cur.execute(
                "SELECT name FROM collections WHERE collection_id = %s",
                (r[3],),
            )
            collection = cur.fetchone()
            notebook["collection_name"] = collection[0] if collection else None
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

    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify notebook ownership
    cur.execute(
        """
        SELECT 1 FROM notebooks
        WHERE notebook_id = %s AND user_id = %s
        """,
        (notebook_id, user_id),
    )
    owned = cur.fetchone()

    if not owned:
        raise HTTPException(status_code=403, detail="Forbidden")

    loaded = load_vectors(notebook_id)
    if loaded:
        _, metadata = loaded
        vectors = metadata
    else:
        vectors = []

    answer = generate_answer(question, vectors, notebook_id)

    # Store chat history
    cur.execute(
        """
        INSERT INTO chat_history (notebook_id, role, content)
        VALUES (%s, %s, %s)
        """,
        (notebook_id, "user", question),
    )
    cur.execute(
        """
        INSERT INTO chat_history (notebook_id, role, content)
        VALUES (%s, %s, %s)
        """,
        (notebook_id, "assistant", answer),
    )

    repo.conn.commit()

    return {"answer": answer}


# --------------------------------------------------
# Collection Chat
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

    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify collection ownership
    cur.execute(
        """
        SELECT name FROM collections
        WHERE collection_id = %s AND user_id = %s
        """,
        (payload.collection_id, user_id),
    )
    owned = cur.fetchone()

    if not owned:
        raise HTTPException(status_code=403, detail="Collection not found or not owned")

    collection_name = owned[0]

    # Generate answer using collection-aware pipeline
    answer = generate_answer(
        question=payload.question,
        collection_id=payload.collection_id,
        user_id=user_id,
    )

    # Store in collection chat history
    try:
        cur.execute(
            """
            INSERT INTO collection_chat_history
            (collection_id, user_id, role, content)
            VALUES (%s, %s, %s, %s)
            """,
            (payload.collection_id, user_id, "user", payload.question),
        )
        cur.execute(
            """
            INSERT INTO collection_chat_history
            (collection_id, user_id, role, content)
            VALUES (%s, %s, %s, %s)
            """,
            (payload.collection_id, user_id, "assistant", answer),
        )
        repo.conn.commit()
    except Exception as e:
        # Table might not exist (backward compatibility)
        print(f"Note: collection_chat_history table not available: {e}")

    # Get notebook count for info
    notebook_ids = get_collection_notebooks(payload.collection_id, user_id)

    return {
        "answer": answer,
        "collection_id": payload.collection_id,
        "collection_name": collection_name,
        "sources": len(notebook_ids),
    }
# --------------------------------------------------
# Delete notebook
# --------------------------------------------------

@app.delete("/notebook/{notebook_id}")
def delete_notebook(
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify ownership
    cur.execute(
        """
        SELECT 1 FROM notebooks
        WHERE notebook_id = %s AND user_id = %s
        """,
        (notebook_id, user_id),
    )
    owned = cur.fetchone()

    if not owned:
        raise HTTPException(status_code=404, detail="Not found")

    # Delete notebook + chat history
    cur.execute(
        "DELETE FROM notebooks WHERE notebook_id = %s",
        (notebook_id,),
    )
    cur.execute(
        "DELETE FROM chat_history WHERE notebook_id = %s",
        (notebook_id,),
    )

    repo.conn.commit()

    # Remove FAISS vectors
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
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        SELECT role, content, created_at
        FROM chat_history
        WHERE notebook_id = %s
          AND notebook_id IN (
              SELECT notebook_id
              FROM notebooks
              WHERE user_id = %s
          )
        ORDER BY created_at ASC
        """,
        (notebook_id, user_id),
    )

    rows = cur.fetchall()

    return [
        {
            "role": r[0],
            "content": r[1],
            "created_at": r[2],
        }
        for r in rows
    ]


# --------------------------------------------------
# Collection Contents
# --------------------------------------------------

@app.get("/collection/{collection_id}/contents")
def get_collection_contents(
    collection_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Get all notebooks and their info in a collection
    """
    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify ownership
    cur.execute(
        """
        SELECT name
        FROM collections
        WHERE collection_id = %s AND user_id = %s
        """,
        (collection_id, user_id),
    )
    owned = cur.fetchone()

    if not owned:
        raise HTTPException(status_code=403, detail="Collection not found or not owned")

    collection_name = owned[0]

    # Get notebooks in collection
    cur.execute(
        """
        SELECT notebook_id, filename, created_at
        FROM notebooks
        WHERE collection_id = %s AND user_id = %s
        ORDER BY created_at
        """,
        (collection_id, user_id),
    )
    notebooks = cur.fetchall()

    notebook_details = []

    for nb in notebooks:
        notebook_id, filename, created_at = nb

        meta_path = BASE_DIR / "data" / "faiss" / f"{notebook_id}.json"
        chunk_count = 0

        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                chunk_count = len(metadata)
            except Exception:
                chunk_count = 0

        notebook_details.append(
            {
                "notebook_id": notebook_id,
                "filename": filename,
                "created_at": created_at,
                "chunk_count": chunk_count,
            }
        )

    return {
        "collection_id": collection_id,
        "collection_name": collection_name,
        "notebooks": notebook_details,
        "total_notebooks": len(notebook_details),
    }
# --------------------------------------------------
# Search Across Collection
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

    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify ownership
    cur.execute(
        """
        SELECT name
        FROM collections
        WHERE collection_id = %s AND user_id = %s
        """,
        (collection_id, user_id),
    )
    owned = cur.fetchone()

    if not owned:
        raise HTTPException(status_code=403, detail="Collection not found or not owned")

    collection_name = owned[0]

    # Perform semantic search
    results = similarity_search(
        question=question,
        collection_id=collection_id,
        user_id=user_id,
        TOP_K=top_k,
    )

    # Format results with preview
    formatted_results = []
    for result in results:
        text_preview = result.get("text", "")
        if len(text_preview) > 300:
            text_preview = text_preview[:300] + "..."

        formatted_results.append(
            {
                "text": result.get("text", ""),
                "preview": text_preview,
                "score": round(result.get("score", 0), 4),
                "notebook_id": result.get("notebook_id", ""),
                "source": result.get("source", "Unknown"),
                "chunk_index": result.get("chunk_index", 0),
            }
        )

    return {
        "collection_id": collection_id,
        "collection_name": collection_name,
        "question": question,
        "total_results": len(results),
        "results": formatted_results,
    }


# --------------------------------------------------
# Collection History
# --------------------------------------------------

@app.get("/collection/{collection_id}/history")
def get_collection_chat_history(
    collection_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Get chat history for a collection
    """
    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify ownership
    cur.execute(
        """
        SELECT name
        FROM collections
        WHERE collection_id = %s AND user_id = %s
        """,
        (collection_id, user_id),
    )
    owned = cur.fetchone()

    if not owned:
        raise HTTPException(status_code=403, detail="Collection not found or not owned")

    collection_name = owned[0]

    # Fetch history (table may or may not exist)
    try:
        cur.execute(
            """
            SELECT role, content, created_at
            FROM collection_chat_history
            WHERE collection_id = %s AND user_id = %s
            ORDER BY created_at ASC
            """,
            (collection_id, user_id),
        )
        rows = cur.fetchall()

        history = [
            {
                "role": r[0],
                "content": r[1],
                "created_at": r[2],
            }
            for r in rows
        ]
    except Exception:
        history = []

    return {
        "collection_id": collection_id,
        "collection_name": collection_name,
        "history": history,
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

    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        INSERT INTO collections (collection_id, name, user_id)
        VALUES (%s, %s, %s)
        """,
        (collection_id, data.name, user_id),
    )

    repo.conn.commit()

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
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        SELECT 
            c.collection_id,
            c.name,
            c.created_at,
            COUNT(n.notebook_id) AS notebook_count
        FROM collections c
        LEFT JOIN notebooks n
          ON c.collection_id = n.collection_id
         AND n.user_id = c.user_id
        WHERE c.user_id = %s
        GROUP BY c.collection_id, c.name, c.created_at
        ORDER BY c.created_at DESC
        """,
        (user_id,),
    )

    rows = cur.fetchall()

    return [
        {
            "collection_id": r[0],
            "name": r[1],
            "created_at": r[2],
            "notebook_count": r[3] or 0,
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
    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify ownership
    cur.execute(
        """
        SELECT 1
        FROM collections
        WHERE collection_id = %s AND user_id = %s
        """,
        (collection_id, user_id),
    )
    owned = cur.fetchone()

    if not owned:
        raise HTTPException(status_code=404, detail="Collection not found")

    # Unlink notebooks from collection
    cur.execute(
        """
        UPDATE notebooks
        SET collection_id = NULL
        WHERE collection_id = %s AND user_id = %s
        """,
        (collection_id, user_id),
    )

    # Delete collection
    cur.execute(
        "DELETE FROM collections WHERE collection_id = %s",
        (collection_id,),
    )

    # Clean up collection chat history (best-effort)
    try:
        cur.execute(
            "DELETE FROM collection_chat_history WHERE collection_id = %s",
            (collection_id,),
        )
    except Exception:
        pass

    repo.conn.commit()

    return {"status": "deleted"}
# --------------------------------------------------
# PODCAST ENDPOINTS
# --------------------------------------------------

@app.post("/podcast/generate")
def generate_podcast(
    data: PodcastRequest,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify notebook ownership
    cur.execute(
        """
        SELECT 1 FROM notebooks
        WHERE notebook_id = %s AND user_id = %s
        """,
        (data.notebook_id, user_id),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=403, detail="Forbidden")

    job_id = str(uuid.uuid4())

    cur.execute(
        """
        INSERT INTO podcast_jobs
        (id, notebook_id, user_id, status, speakers)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (job_id, data.notebook_id, user_id, "pending", data.speakers),
    )
    repo.conn.commit()

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
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        SELECT status, result
        FROM podcast_jobs
        WHERE id = %s AND user_id = %s
        """,
        (job_id, user_id),
    )
    row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "status": row[0],
        "result": row[1] if row[0] in ("script_ready", "done") else None,
    }


@app.get("/podcast/latest/{notebook_id}")
def get_latest_podcast(
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        """
        SELECT id, result, speakers
        FROM podcast_jobs
        WHERE notebook_id = %s
          AND user_id = %s
          AND status IN ('script_ready', 'done')
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (notebook_id, user_id),
    )
    row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="No podcast found")

    return {
        "job_id": row[0],
        "result": row[1],
        "speakers": row[2],
    }


@app.get("/podcast/audio/{job_id}")
def get_podcast_audio(job_id: str):
    repo = get_repo()
    cur = repo.conn.cursor()

    cur.execute(
        "SELECT audio_path FROM podcast_jobs WHERE id = %s",
        (job_id,),
    )
    row = cur.fetchone()

    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Audio not ready")

    audio_path = Path(row[0])
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


# --------------------------------------------------
# REMOVE NOTEBOOK FROM COLLECTION
# --------------------------------------------------

@app.delete("/collection/{collection_id}/notebook/{notebook_id}")
def remove_notebook_from_collection(
    collection_id: str,
    notebook_id: str,
    user_id: str = Depends(get_current_user_id),
):
    repo = get_repo()
    cur = repo.conn.cursor()

    # Verify collection ownership
    cur.execute(
        """
        SELECT 1 FROM collections
        WHERE collection_id = %s AND user_id = %s
        """,
        (collection_id, user_id),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=403, detail="Collection not found or not owned")

    # Verify notebook ownership
    cur.execute(
        """
        SELECT 1 FROM notebooks
        WHERE notebook_id = %s AND user_id = %s
        """,
        (notebook_id, user_id),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail="Notebook not found")

    # Remove notebook from collection
    cur.execute(
        """
        UPDATE notebooks
        SET collection_id = NULL
        WHERE notebook_id = %s AND collection_id = %s
        """,
        (notebook_id, collection_id),
    )
    repo.conn.commit()

    return {
        "status": "removed",
        "notebook_id": notebook_id,
        "collection_id": collection_id,
    }
