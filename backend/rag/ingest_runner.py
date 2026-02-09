from pathlib import Path
from rag.pdf_reader import read_pdf
from rag.text_splitter import split_text
from rag.embedder import embed
from rag.vector_store import save_vectors
from rag.ingest import ingest_document

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"


def run_ingestion(
    notebook_id: str,
    user_id: str,
    collection_id: str | None,
    filename: str,
):
    pdf_path = UPLOAD_DIR / f"{notebook_id}.pdf"
    if not pdf_path.exists():
        raise RuntimeError("PDF file missing")

    text = read_pdf(pdf_path)
    if not text.strip():
        raise RuntimeError("Empty PDF")

    chunks = split_text(text)
    embeddings = embed(chunks)

    vectors = [{"text": c, "embedding": e} for c, e in zip(chunks, embeddings)]
    save_vectors(notebook_id, vectors)

    ingest_document(
        text=text,
        document_id=notebook_id,
        user_id=user_id,
        collection_id=collection_id,
        filename=filename,
    )

