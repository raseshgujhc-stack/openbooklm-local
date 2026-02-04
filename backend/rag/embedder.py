# file: embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

# Use same model for both documents and queries
model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # Explicit model name

def embed(texts):
    """
    Embed a list of texts (for documents)
    Normalize embeddings for cosine similarity
    """
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,  # Crucial for cosine similarity
        show_progress_bar=False,
        batch_size=32
    )
    return embeddings.tolist()

def embed_query(text):
    """
    Embed a single query string
    Same model, same normalization
    """
    embedding = model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embedding[0].tolist()

def embed_texts(texts: list[str]):
    """
    Batch embed texts.
    Returns List[List[float]]
    """
    return embed(texts)  # Use the same function
