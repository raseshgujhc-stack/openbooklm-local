# file: migrate_db.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "notebooks.db"

def backup_database():
    """Create a backup of the database"""
    import shutil
    import datetime
    
    backup_path = DB_PATH.parent / f"notebooks_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    shutil.copy2(DB_PATH, backup_path)
    print(f"Database backed up to: {backup_path}")
    return backup_path

def check_and_fix_schema():
    """Check and fix any schema issues"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    print("Checking database schema...")
    
    # Check if collections table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='collections'")
    if not cur.fetchone():
        print("Creating collections table...")
        cur.execute("""
            CREATE TABLE collections (
                collection_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    # Check if notebooks table has collection_id column
    cur.execute("PRAGMA table_info(notebooks)")
    columns = [col[1] for col in cur.fetchall()]
    
    if 'collection_id' not in columns:
        print("Adding collection_id column to notebooks table...")
        cur.execute("ALTER TABLE notebooks ADD COLUMN collection_id TEXT")
    
    # Check if collection_chat_history table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='collection_chat_history'")
    if not cur.fetchone():
        print("Creating collection_chat_history table...")
        cur.execute("""
            CREATE TABLE collection_chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    # Create indexes
    print("Creating indexes...")
    indexes = [
        ("idx_notebooks_user", "CREATE INDEX idx_notebooks_user ON notebooks(user_id)"),
        ("idx_notebooks_collection", "CREATE INDEX idx_notebooks_collection ON notebooks(collection_id)"),
        ("idx_collections_user", "CREATE INDEX idx_collections_user ON collections(user_id)"),
        ("idx_chat_history_notebook", "CREATE INDEX idx_chat_history_notebook ON chat_history(notebook_id)"),
        ("idx_collection_chat_collection", "CREATE INDEX idx_collection_chat_collection ON collection_chat_history(collection_id)"),
        ("idx_collection_chat_user", "CREATE INDEX idx_collection_chat_user ON collection_chat_history(user_id)")
    ]
    
    for idx_name, idx_sql in indexes:
        cur.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND name='{idx_name}'")
        if not cur.fetchone():
            print(f"Creating index: {idx_name}")
            cur.execute(idx_sql)
    
    conn.commit()
    conn.close()
    print("Database schema updated successfully!")

def main():
    print("Database Migration Tool")
    print("=" * 50)
    
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("Creating new database...")
        from db import init_db
        init_db()
        return
    
    # Backup first
    backup_path = backup_database()
    
    # Fix schema
    check_and_fix_schema()
    
    print("\nMigration complete!")
    print(f"Original database backed up to: {backup_path}")

if __name__ == "__main__":
    main()
