import sqlite3
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "notebooks.db"


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    # -----------------------------
    # Users table
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # -----------------------------
    # Collections table (NEW)
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS collections (
            collection_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # -----------------------------
    # Notebooks table (updated with collection_id)
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notebooks (
            notebook_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            user_id TEXT NOT NULL,
            collection_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE SET NULL
        )
    """)

    # -----------------------------
    # Chat history table (single notebook chat)
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            notebook_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (notebook_id) REFERENCES notebooks(notebook_id) ON DELETE CASCADE
        )
    """)

    # -----------------------------
    # Collection chat history table (NEW)
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS collection_chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (collection_id) REFERENCES collections(collection_id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # -----------------------------
    # Podcast jobs table
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS podcast_jobs (
            id TEXT PRIMARY KEY,
            notebook_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            language TEXT,
            speakers INTEGER,
            status TEXT,           -- pending | running | done | error
            result TEXT,           -- podcast script
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            audio_path TEXT,
            FOREIGN KEY (notebook_id) REFERENCES notebooks(notebook_id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # -----------------------------
    # Podcasts table (optional, for storing completed podcasts)
    # -----------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS podcasts (
            id TEXT PRIMARY KEY,
            notebook_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            title TEXT,
            language TEXT DEFAULT 'en',
            speakers INTEGER DEFAULT 2,
            script TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (notebook_id) REFERENCES notebooks(notebook_id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # -----------------------------
    # Create indexes for better performance
    # -----------------------------
    
    # Users indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
    
    # Notebooks indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_notebooks_user ON notebooks(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_notebooks_collection ON notebooks(collection_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_notebooks_created ON notebooks(created_at)")
    
    # Collections indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_collections_user ON collections(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_collections_created ON collections(created_at)")
    
    # Chat history indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_notebook ON chat_history(notebook_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_created ON chat_history(created_at)")
    
    # Collection chat history indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_collection_chat_collection ON collection_chat_history(collection_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_collection_chat_user ON collection_chat_history(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_collection_chat_created ON collection_chat_history(created_at)")
    
    # Podcast jobs indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_podcast_jobs_user ON podcast_jobs(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_podcast_jobs_notebook ON podcast_jobs(notebook_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_podcast_jobs_status ON podcast_jobs(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_podcast_jobs_created ON podcast_jobs(created_at)")
    
    # Podcasts indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_podcasts_user ON podcasts(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_podcasts_notebook ON podcasts(notebook_id)")

    conn.commit()
    conn.close()


def init_chat_table():
    """Initialize chat tables (for backward compatibility)"""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            notebook_id TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Also ensure collection_chat_history exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS collection_chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()


def migrate_database():
    """
    Migration function to update existing database to new schema
    """
    conn = get_db()
    cur = conn.cursor()
    
    # Check if old chat_history table has 'message' column instead of 'content'
    try:
        cur.execute("PRAGMA table_info(chat_history)")
        columns = [col[1] for col in cur.fetchall()]
        
        if 'message' in columns and 'content' not in columns:
            # Migrate from 'message' to 'content'
            print("Migrating chat_history table from 'message' to 'content' column...")
            
            # Create new table with correct schema
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    notebook_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Copy data from old table
            cur.execute("""
                INSERT INTO chat_history_new (id, notebook_id, role, content, created_at)
                SELECT id, notebook_id, role, message, created_at
                FROM chat_history
            """)
            
            # Drop old table
            cur.execute("DROP TABLE chat_history")
            
            # Rename new table
            cur.execute("ALTER TABLE chat_history_new RENAME TO chat_history")
            
            print("Migration completed successfully!")
    except Exception as e:
        print(f"Migration check: {e}")
    
    # Ensure collections table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS collections (
            collection_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Ensure notebooks table has collection_id column
    try:
        cur.execute("PRAGMA table_info(notebooks)")
        columns = [col[1] for col in cur.fetchall()]
        
        if 'collection_id' not in columns:
            print("Adding collection_id column to notebooks table...")
            cur.execute("ALTER TABLE notebooks ADD COLUMN collection_id TEXT")
    except Exception as e:
        print(f"Column check: {e}")
    
    # Ensure collection_chat_history table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS collection_chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()


# Initialize DB on import
init_db()
# Run migration to ensure schema is up-to-date
migrate_database()
