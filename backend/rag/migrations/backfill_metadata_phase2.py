import sqlite3
import json
from pathlib import Path
import hashlib

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data" / "notebooks.db"
FAISS_DIR = BASE_DIR / "data" / "faiss"

def reconstruct_text(notebook_id):
    meta_path = FAISS_DIR / f"{notebook_id}.json"
    if not meta_path.exists():
        return None

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return "\n".join(m["text"] for m in metadata)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
SELECT document_id FROM document_metadata
WHERE word_count IS NULL
""")

rows = cursor.fetchall()

for (doc_id,) in rows:
    text = reconstruct_text(doc_id)
    if not text:
        continue

    word_count = len(text.split())
    file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    cursor.execute("""
    UPDATE document_metadata
    SET word_count = ?, file_hash = ?
    WHERE document_id = ?
    """, (word_count, file_hash, doc_id))

conn.commit()
conn.close()

print("✅ Phase 2 complete: Deterministic metadata filled")

