# file: chunking.py
import re
from typing import List

def clean_text(text: str) -> str:
    # More aggressive cleaning for legal docs
    text = re.sub(r'\s+', ' ', text)  # Replace all whitespace with single space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_into_chunks(
    text: str,
    max_chars: int = 800,  # Smaller for precision
    overlap: int = 100,
) -> List[str]:
    """
    Legal-aware chunking that preserves document structure
    """
    text = clean_text(text)
    
    # Split by sentences first for legal docs
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed max_chars
        if current_length + sentence_length > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            words = current_chunk.split()
            overlap_words = words[-20:] if len(words) >= 20 else words
            current_chunk = " ".join(overlap_words) + " " + sentence
            current_length = len(current_chunk)
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_length += sentence_length
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks
