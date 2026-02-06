# backend/db/postgres_repo.py

import psycopg2
from psycopg2.extras import execute_values

class PostgresMetadataRepository:
    def __init__(self, dsn):
        self.conn = psycopg2.connect(dsn)

    def insert_document(self, data: dict):
        """
        Insert metadata safely.
        Only inserts keys that exist in document_metadata table.
        """
        columns = []
        values = []

        for key, value in data.items():
            columns.append(key)
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

