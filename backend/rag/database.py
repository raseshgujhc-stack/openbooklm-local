import sqlite3

conn = sqlite3.connect("data/notebooks.db")
cursor = conn.cursor()

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_meta_collection
ON document_metadata (collection_id, user_id);
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_meta_doc
ON document_metadata (document_id);
""")

conn.commit()
print("✅ Indexes created")


cursor.execute("""
INSERT OR IGNORE INTO document_metadata (
    document_id, user_id, collection_id, ingested_at
)
SELECT notebook_id, user_id, collection_id, CURRENT_TIMESTAMP
FROM notebooks;
""")

conn.commit()
conn.close()

print("✅ Backfilled basic metadata")
