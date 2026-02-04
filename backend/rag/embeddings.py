# rag/embeddings.py
from sentence_transformers import SentenceTransformer

# BEST CPU embedding model today
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Returns list of embeddings.
    CPU friendly, fast, stable.
    """
    embeddings = _model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.tolist()
