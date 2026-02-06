import sqlite3
import psycopg2

SQLITE_DB = "/home/ubuntu/openbooklm-local/backend/data/notebooks.db"
PG_DSN = "dbname=notebooklm user=notebook_user password=GujHC@123 host=localhost"

sqlite_conn = sqlite3.connect(SQLITE_DB)
sqlite_conn.row_factory = sqlite3.Row
pg_conn = psycopg2.connect(PG_DSN)

def migrate_table(table_name):
    sc = sqlite_conn.cursor()
    pc = pg_conn.cursor()

    sc.execute(f"SELECT * FROM {table_name}")
    rows = sc.fetchall()

    if not rows:
        print(f"[SKIP] {table_name} empty")
        return

    columns = rows[0].keys()
    cols = ",".join(columns)
    placeholders = ",".join(["%s"] * len(columns))

    for row in rows:
        values = [row[col] for col in columns]
        pc.execute(
            f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})",
            values
        )

    pg_conn.commit()
    print(f"[OK] Migrated {len(rows)} rows from {table_name}")

TABLES_IN_ORDER = [
    "users",
    "collections",
    "notebooks",
    "document_metadata",
    "chat_history",
    "collection_chat_history",
    "podcasts",
    "podcast_jobs",
]

for table in TABLES_IN_ORDER:
    migrate_table(table)

sqlite_conn.close()
pg_conn.close()

