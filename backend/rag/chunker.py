# rag/chunker.py

def chunk_text(text, chunk_size=800, overlap=150):
    """
    Semantic-safe chunking for legal documents.
    """

    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]

    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) <= chunk_size:
            current += " " + p
        else:
            chunks.append(current.strip())
            current = current[-overlap:] + " " + p

    if current.strip():
        chunks.append(current.strip())

    return chunks
