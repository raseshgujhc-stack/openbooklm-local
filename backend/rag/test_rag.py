# test_rag.py
from rag.embedder import embed, embed_query
from rag.chunking import split_into_chunks
from rag.similarity import similarity_search
import numpy as np

def test_rag_pipeline():
    # Test document
    doc_text = "The contract stipulates that payment must be made within 30 days of invoice receipt. Late payments incur a 5% monthly penalty."
    
    # Chunk
    chunks = split_into_chunks(doc_text)
    print(f"Chunks: {len(chunks)}")
    
    # Embed chunks
    chunk_embeddings = embed(chunks)
    
    # Create vectors
    vectors = [{"text": chunk, "embedding": emb} for chunk, emb in zip(chunks, chunk_embeddings)]
    
    # Test query
    query = "What is the penalty for late payment?"
    
    # Search
    results = similarity_search(query, vectors, None, TOP_K=3)
    
    print("\nSearch Results:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Score: {result['score']:.3f}):")
        print(result['text'][:200])
    
    # Check if we found the right info
    if results and results[0]['score'] > 0.7:
        print("\n✅ RAG pipeline working correctly!")
    else:
        print("\n❌ RAG pipeline needs adjustment")

if __name__ == "__main__":
    test_rag_pipeline()
