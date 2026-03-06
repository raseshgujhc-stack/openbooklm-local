# backend/db/postgres_repo.py

import psycopg2
from psycopg2.extras import Json


JSON_FIELDS = {
    "act_names",
    "primary_topics",
    "secondary_topics",
    "keywords",
    "referenced_laws",
    "section_types",
    "cited_cases",
    "cited_courts",
    "cited_acts",
    "jurisdiction",
    "field_confidence",
    "extraction_notes",
    "bench",
    "judge_name",
    "inferred_subjects",
    "chapter_titles",
    "toc_entries",
    "act_alias_hits",
    "structure_hints",
}


class PostgresMetadataRepository:
    def __init__(self, dsn):
        self.conn = psycopg2.connect(dsn)

    def insert_document(self, data: dict):
        """
        Insert metadata safely.
        Automatically serializes JSONB fields.
        """
        columns = []
        values = []

        for key, value in data.items():
            columns.append(key)

            if key in JSON_FIELDS and value is not None:
                values.append(Json(value))   # ✅ FIX
            else:
                values.append(value)

        cols_sql = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(values))

        sql = f"""
            INSERT INTO document_metadata ({cols_sql})
            VALUES ({placeholders})
        """

        with self.conn.cursor() as cur:
            cur.execute(sql, values)
            self.conn.commit()

    def insert_book_document(self, data: dict):
        """
        Insert book-focused metadata.
        Remark: kept separate from judicial metadata so book schema can evolve independently.
        """
        columns = []
        values = []

        for key, value in data.items():
            columns.append(key)
            if key in JSON_FIELDS and value is not None:
                values.append(Json(value))
            else:
                values.append(value)

        cols_sql = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(values))
        sql = f"""
            INSERT INTO book_metadata ({cols_sql})
            VALUES ({placeholders})
            ON CONFLICT (document_id) DO UPDATE
            SET {", ".join([f"{c}=EXCLUDED.{c}" for c in columns if c != "document_id"])}
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, values)
            self.conn.commit()

    def upsert_book_section_rows(self, document_id: str, rows: list[dict]):
        """
        Replace section index rows for one document.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM book_section_index WHERE document_id = %s",
                (document_id,),
            )
            if rows:
                insert_sql = """
                    INSERT INTO book_section_index (
                        document_id, filename, user_id, collection_id,
                        act_canonical, section_code, parent_section_code, section_type, section_title,
                        chunk_index, text_preview
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, act_canonical, section_code, chunk_index)
                    DO UPDATE SET
                        filename = EXCLUDED.filename,
                        user_id = EXCLUDED.user_id,
                        collection_id = EXCLUDED.collection_id,
                        parent_section_code = EXCLUDED.parent_section_code,
                        section_type = EXCLUDED.section_type,
                        section_title = EXCLUDED.section_title,
                        text_preview = EXCLUDED.text_preview
                """
                for row in rows:
                    cur.execute(
                        insert_sql,
                        (
                            document_id,
                            row.get("filename"),
                            row.get("user_id"),
                            row.get("collection_id"),
                            row.get("act_canonical"),
                            row.get("section_code"),
                            row.get("parent_section_code"),
                            row.get("section_type"),
                            row.get("section_title"),
                            row.get("chunk_index"),
                            row.get("text_preview"),
                        ),
                    )
            self.conn.commit()

    def fetch_by_collection(self, collection_id: str, user_id: str):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT *
                FROM document_metadata
                WHERE collection_id = %s AND user_id = %s
            """, (collection_id, user_id))
            return cur.fetchall()

    def fetch_by_document(self, document_id: str):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT *
                FROM document_metadata
                WHERE document_id = %s
            """, (document_id,))
            return cur.fetchone()
