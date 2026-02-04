from rag.ingest import extract_llm_metadata
import sqlite3
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "notebooks.db"
FAISS_DIR = BASE_DIR / "data" / "faiss"

def reconstruct_text(notebook_id):
    meta_path = FAISS_DIR / f"{notebook_id}.json"
    if not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return "\n".join(m["text"] for m in metadata[:5])  # only first chunks

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
SELECT document_id FROM document_metadata
WHERE document_type IS NULL
LIMIT 10
""")

rows = cursor.fetchall()

for (doc_id,) in rows:
    text = reconstruct_text(doc_id)
    if not text:
        continue

    llm_meta = extract_llm_metadata(text)

    cursor.execute("""
    UPDATE document_metadata
    SET
      domain = ?,
      document_type = ?,
      case_stage = ?,
      petition_type = ?,
      act_name = ?,
      court_level = ?
    WHERE document_id = ?
    """, (
        llm_meta.get("domain"),
        llm_meta.get("document_type"),
        llm_meta.get("case_stage"),
        llm_meta.get("petition_type"),
        llm_meta.get("act_name"),
        llm_meta.get("court_level"),
        doc_id
    ))

conn.commit()
conn.close()

print("✅ Phase 3 batch complete")
