# file: config.py
RAG_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 100,
    "top_k": 5,
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "similarity_threshold": 0.7,  # Minimum similarity score
    "max_context_length": 4000,   # Max tokens for LLM context
}
