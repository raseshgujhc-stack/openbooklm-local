import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data" / "notebooks.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Insert metadata rows ONLY if missing
cursor.execute("""
INSERT OR IGNORE INTO document_metadata (
    document_id,
    user_id,
    collection_id,
    filename,
    created_at,
    ingested_at
)
SELECT
    notebook_id,
    user_id,
    collection_id,
    filename,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
FROM notebooks
""")

conn.commit()
conn.close()

print("✅ Phase 1 complete: Existing documents registered")

