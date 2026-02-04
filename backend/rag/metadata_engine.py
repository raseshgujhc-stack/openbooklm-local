# rag/metadata_engine.py

import sqlite3

DB_PATH = "data/notebooks.db"

def handle_metadata_query(question, collection_id, user_id, notebook_id=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    q = question.lower()

    # ------------------------------
    # Collection-level metadata
    # ------------------------------
    if collection_id and user_id:

        if "order date" in q:
            cursor.execute("""
            SELECT case_number, order_date
            FROM document_metadata
            WHERE collection_id = ? AND user_id = ?
            """, (collection_id, user_id))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return "Order date information not available."

            return "\n".join(
                f"Case: {r[0] or 'N/A'}, Order Date: {r[1] or 'N/A'}"
                for r in rows
            )

        if "how many" in q or "count" in q:
            cursor.execute("""
            SELECT COUNT(*) FROM document_metadata
            WHERE collection_id = ? AND user_id = ?
            """, (collection_id, user_id))

            count = cursor.fetchone()[0]
            conn.close()
            return f"Total documents: {count}"

    # ------------------------------
    # Single-document metadata
    # ------------------------------
    if notebook_id:
        cursor.execute("""
        SELECT page_count, word_count, document_type
        FROM document_metadata
        WHERE document_id = ?
        """, (notebook_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return "Metadata not available for this document."

        return (
            f"Pages: {row[0] or 'N/A'}, "
            f"Words: {row[1] or 'N/A'}, "
            f"Document Type: {row[2] or 'Unknown'}"
        )

    conn.close()
    return "Metadata query not supported."

def handle_metadata_intent(intent, collection_id, user_id, notebook_id=None):
    """
    Handles metadata queries using structured intent.
    """

    import sqlite3
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    DB_PATH = BASE_DIR / "data" / "notebooks.db"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    op = intent.get("operation")
    entities = intent.get("entities", {})
    filters = intent.get("filters", {})

    # ===============================
    # LIST CASES
    # ===============================
    if op == "list" and entities.get("case"):

        cursor.execute("""
        SELECT case_number, document_type, order_date
        FROM document_metadata
        WHERE collection_id = ? AND user_id = ?
        """, (collection_id, user_id))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No cases found in this collection."

        return "\n".join(
            f"Case: {r[0] or 'N/A'}, Type: {r[1] or 'Unknown'}, Order Date: {r[2] or 'N/A'}"
            for r in rows
        )

    # ===============================
    # COUNT DOCUMENTS
    # ===============================
    if op == "count":
        cursor.execute("""
        SELECT COUNT(*) FROM document_metadata
        WHERE collection_id = ? AND user_id = ?
        """, (collection_id, user_id))

        count = cursor.fetchone()[0]
        conn.close()
        return f"Total documents: {count}"

    conn.close()
    return "Metadata intent recognized but not yet supported."

