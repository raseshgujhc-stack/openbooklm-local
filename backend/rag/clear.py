from pathlib import Path
from rag.vector_store import delete_vectors

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"


def hard_delete_notebook(cur, notebook_id: str):
    """
    Completely removes a notebook and ALL related data.
    Must be called inside a transaction.
    """

    # Chat + RAG metadata
    cur.execute("DELETE FROM chat_history WHERE notebook_id = %s", (notebook_id,))
    cur.execute(
        "DELETE FROM document_metadata WHERE document_id = %s",
        (notebook_id,),
    )
    # Remark: keep book metadata lifecycle aligned with notebook deletion.
    cur.execute(
        "DELETE FROM book_metadata WHERE document_id = %s",
        (notebook_id,),
    )
    cur.execute(
        "DELETE FROM book_section_index WHERE document_id = %s",
        (notebook_id,),
    )

    # Ingestion + derived artifacts
    cur.execute("DELETE FROM ingest_jobs WHERE notebook_id = %s", (notebook_id,))
    cur.execute("DELETE FROM podcast_jobs WHERE notebook_id = %s", (notebook_id,))
    cur.execute("DELETE FROM podcasts WHERE notebook_id = %s", (notebook_id,))

    # Notebook itself
    cur.execute("DELETE FROM notebooks WHERE notebook_id = %s", (notebook_id,))

    # ---- filesystem cleanup (best effort, after commit) ----
    delete_vectors(notebook_id)

    pdf_path = UPLOAD_DIR / f"{notebook_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()
