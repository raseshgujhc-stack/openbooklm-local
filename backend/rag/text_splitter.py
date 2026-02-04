def split_text(text, max_len=1200):
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) < max_len:
            current += "\n\n" + p
        else:
            chunks.append(current.strip())
            current = p

    if current.strip():
        chunks.append(current.strip())

    return chunks
