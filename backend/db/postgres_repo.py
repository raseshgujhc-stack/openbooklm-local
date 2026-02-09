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

