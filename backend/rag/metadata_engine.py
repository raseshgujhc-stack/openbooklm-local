# rag/metadata_engine.py
"""
Structured metadata answer engine.

Handles user questions that can be answered directly from document_metadata
without running full semantic RAG generation.
"""

from db import get_repo
from rag.model_router import qwen_summary


def _is_book_role(role: str | None) -> bool:
    return (role or "").lower() in {"referencebook", "book"}


def _format_single_doc_row(row):
    (
        filename,
        doc_type,
        role,
        page_count,
        word_count,
        court_name,
        order_date,
        book_title,
        section_count,
        article_count,
        part_count,
        schedule_count,
        chapter_count,
    ) = row

    # Remark: render book metadata from book schema when document is classified as book.
    if _is_book_role(role) or (doc_type or "").lower() == "book":
        return (
            "Document in scope:\n\n"
            f"- {book_title or filename or 'Untitled Book'} | Book | "
            f"Pages: {page_count or 'N/A'} | Sections: {section_count or 0} | "
            f"Articles: {article_count or 0} | Parts: {part_count or 0} | "
            f"Schedules: {schedule_count or 0} | "
            f"Chapters: {chapter_count or 0}"
        )

    return (
        "Document in scope:\n\n"
        f"- {filename} | {doc_type or 'Unknown'} | {court_name or 'Unknown Court'} | {order_date or 'No Date'}"
    )


def is_metadata_question(question: str) -> bool:
    """
    Remark: broad metadata classifier so we don't hardcode each exact phrasing.
    """
    q = (question or "").lower()
    if not q.strip():
        return False

    direct_markers = [
        "how many",
        "count",
        "list",
        "show",
        "which document",
        "documents and pages",
        "page count",
        "total pages",
        "same court",
        "same judge",
        "order date",
        "document type",
        "case number",
    ]
    if any(m in q for m in direct_markers):
        return True

    # Generic signal-based fallback.
    entities = ["document", "documents", "pdf", "pages", "court", "judge", "date", "metadata", "case"]
    operators = ["how many", "count", "list", "show", "same", "which", "what are", "tell me"]
    return any(e in q for e in entities) and any(o in q for o in operators)


def _fetch_scope_rows(cur, collection_id, user_id, notebook_id=None, limit=200):
    if notebook_id and not collection_id:
        cur.execute(
            """
            SELECT
                dm.document_id,
                dm.filename,
                dm.document_role,
                dm.document_type,
                dm.page_count,
                dm.word_count,
                dm.court_name,
                dm.order_date,
                dm.case_number,
                dm.case_type,
                dm.judge_name,
                bm.title,
                bm.section_count,
                bm.article_count,
                bm.part_count,
                bm.schedule_count,
                bm.chapter_count
            FROM document_metadata dm
            LEFT JOIN book_metadata bm ON bm.document_id = dm.document_id
            WHERE dm.document_id = %s
            """,
            (notebook_id,),
        )
        return cur.fetchall()

    if not collection_id or not user_id:
        return []

    cur.execute(
        """
        SELECT
            dm.document_id,
            dm.filename,
            dm.document_role,
            dm.document_type,
            dm.page_count,
            dm.word_count,
            dm.court_name,
            dm.order_date,
            dm.case_number,
            dm.case_type,
            dm.judge_name,
            bm.title,
            bm.section_count,
            bm.article_count,
            bm.part_count,
            bm.schedule_count,
            bm.chapter_count
        FROM document_metadata dm
        LEFT JOIN book_metadata bm ON bm.document_id = dm.document_id
        WHERE dm.collection_id = %s
          AND (dm.user_id = %s OR dm.user_id IS NULL)
        ORDER BY dm.created_at DESC NULLS LAST
        LIMIT %s
        """,
        (collection_id, user_id, int(limit)),
    )
    return cur.fetchall()


def answer_from_metadata_context(question, collection_id, user_id, notebook_id=None):
    """
    Remark: generic metadata answering layer.
    Uses all available structured rows in scope instead of one-off hardcoded patterns.
    """
    if not is_metadata_question(question):
        return None

    repo = get_repo()
    conn = repo.conn
    cur = conn.cursor()
    rows = _fetch_scope_rows(cur, collection_id=collection_id, user_id=user_id, notebook_id=notebook_id)
    if not rows:
        return "Metadata is not available for this scope."

    # Keep prompt compact while still covering mixed book/judgment schema.
    compact_rows = []
    for r in rows[:120]:
        (
            doc_id, filename, role, doc_type, page_count, word_count, court_name,
            order_date, case_number, case_type, judge_name, title, section_count, article_count, part_count, schedule_count, chapter_count
        ) = r
        compact_rows.append(
            {
                "document_id": doc_id,
                "name": title if _is_book_role(role) else filename,
                "role": role,
                "type": doc_type,
                "page_count": page_count,
                "word_count": word_count,
                "court": court_name,
                "order_date": str(order_date) if order_date else None,
                "case_number": case_number,
                "case_type": case_type,
                "judges": judge_name,
                "section_count": section_count,
                "article_count": article_count,
                "part_count": part_count,
                "schedule_count": schedule_count,
                "chapter_count": chapter_count,
            }
        )

    prompt = f"""
You are a strict metadata analyst.

Rules:
- Answer ONLY from the provided metadata rows.
- If the answer is not determinable from metadata, say: "Not determinable from metadata."
- Do not fabricate names, counts, page ranges, or exhibit numbers.
- Prefer concise factual output and distributions where relevant.

User question:
{question}

Metadata rows:
{compact_rows}

Answer:
"""
    try:
        answer = qwen_summary(prompt=prompt, max_tokens=320, temperature=0.1).strip()
        return answer or "Not determinable from metadata."
    except Exception:
        return None


def handle_metadata_query(question, collection_id, user_id, notebook_id=None):
    repo = get_repo()
    conn = repo.conn
    cur = conn.cursor()

    q = question.lower()

    # ------------------------------
    # Collection-level metadata
    # ------------------------------
    if collection_id and user_id:

        if "order date" in q:
            cur.execute("""
                SELECT case_number, order_date
                FROM document_metadata
                WHERE collection_id = %s AND user_id = %s
            """, (collection_id, user_id))

            rows = cur.fetchall()

            if not rows:
                return "Order date information not available."

            return "\n".join(
                f"Case: {r[0] or 'N/A'}, Order Date: {r[1] or 'N/A'}"
                for r in rows
            )

        if "how many" in q or "count" in q:
            cur.execute("""
                SELECT COUNT(*)
                FROM document_metadata
                WHERE collection_id = %s AND user_id = %s
            """, (collection_id, user_id))

            count = cur.fetchone()[0]
            return f"Total documents: {count}"

    # ------------------------------
    # Single-document metadata
    # ------------------------------
    if notebook_id:
        cur.execute("""
            SELECT page_count, word_count, document_type
            FROM document_metadata
            WHERE document_id = %s
        """, (notebook_id,))

        row = cur.fetchone()

        if not row:
            return "Metadata not available for this document."

        return (
            f"Pages: {row[0] or 'N/A'}, "
            f"Words: {row[1] or 'N/A'}, "
            f"Document Type: {row[2] or 'Unknown'}"
        )

    return "Metadata query not supported."


def handle_metadata_intent(intent, collection_id, user_id, notebook_id=None):
    """
    Handles metadata queries using structured intent.
    """

    repo = get_repo()
    conn = repo.conn
    cur = conn.cursor()

    op = intent.get("operation")
    entities = intent.get("entities", {})

    def collection_scope_sql(alias: str = ""):
        # Include global notebooks (user_id IS NULL) alongside user-owned docs.
        prefix = f"{alias}." if alias else ""
        return """
            {prefix}collection_id = %s
            AND ({prefix}user_id = %s OR {prefix}user_id IS NULL)
        """.format(prefix=prefix), (collection_id, user_id)

    # ===============================
    # SAME COURT CHECK
    # ===============================
    if op == "same_court":
        if notebook_id and not collection_id:
            cur.execute(
                """
                SELECT court_name
                FROM document_metadata
                WHERE document_id = %s
                """,
                (notebook_id,),
            )
            row = cur.fetchone()
            if not row:
                return "Court information is not available for this document."
            return f"Single document scope: {row[0] or 'Court not captured'}."

        if not collection_id or not user_id:
            return "Metadata scope not available."

        where_sql, params = collection_scope_sql()
        cur.execute(
            f"""
            SELECT COALESCE(court_name, 'Unknown Court') AS court_name, COUNT(*)
            FROM document_metadata
            WHERE {where_sql}
              AND document_role = 'Judicial'
            GROUP BY COALESCE(court_name, 'Unknown Court')
            ORDER BY COUNT(*) DESC
            """,
            params,
        )
        rows = cur.fetchall()
        if not rows:
            return "No judicial documents found for court comparison in this collection."

        if len(rows) == 1:
            return f"Yes. All documents are from the same court: {rows[0][0]}."

        details = "\n".join([f"- {r[0]}: {r[1]} document(s)" for r in rows[:10]])
        return (
            "No. Documents are from multiple courts.\n\n"
            "Court distribution:\n"
            f"{details}"
        )

    # ===============================
    # SAME JUDGE CHECK
    # ===============================
    if op == "same_judge":
        if not collection_id or not user_id:
            return "Metadata scope not available."

        where_sql, params = collection_scope_sql(alias="dm")
        # Remark: judge_name is a JSON array in this schema.
        cur.execute(
            f"""
            SELECT
                COALESCE(j.value::text, '"Unknown Judge"') AS judge_name,
                COUNT(DISTINCT dm.document_id) AS doc_count
            FROM document_metadata dm
            LEFT JOIN LATERAL jsonb_array_elements_text(COALESCE(dm.judge_name, '[]'::jsonb)) j(value) ON TRUE
            WHERE {where_sql}
              AND dm.document_role = 'Judicial'
            GROUP BY COALESCE(j.value::text, '"Unknown Judge"')
            ORDER BY doc_count DESC
            """,
            params,
        )
        rows = cur.fetchall()
        if not rows:
            return "No judicial documents found for judge comparison in this collection."

        # Normalize quoted text from ::text representation
        normalized = []
        for name, count in rows:
            clean_name = (name or "Unknown Judge").strip('"')
            normalized.append((clean_name, int(count or 0)))

        non_unknown = [r for r in normalized if r[0].lower() != "unknown judge"]
        if len(non_unknown) == 1:
            return f"Yes. All documents are by the same judge: {non_unknown[0][0]}."

        details = "\n".join([f"- {r[0]}: {r[1]} document(s)" for r in normalized[:12]])
        return (
            "No. Documents involve multiple judges.\n\n"
            "Judge distribution:\n"
            f"{details}"
        )

    # ===============================
    # COUNT DOCUMENTS
    # ===============================
    if op == "count":
        if notebook_id and not collection_id:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT document_id FROM document_metadata WHERE document_id = %s
                    UNION
                    SELECT document_id FROM book_metadata WHERE document_id = %s
                ) x
                """,
                (notebook_id, notebook_id),
            )
            count = cur.fetchone()[0]
            return f"Total documents in this scope: {count}"

        if not collection_id or not user_id:
            return "Metadata scope not available."

        where_sql_dm, params_dm = collection_scope_sql()
        where_sql_bm, params_bm = collection_scope_sql()
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT document_id
                FROM document_metadata
                WHERE {where_sql_dm}
                UNION
                SELECT document_id
                FROM book_metadata
                WHERE {where_sql_bm}
            ) x
            """,
            params_dm + params_bm,
        )

        count = cur.fetchone()[0]
        return f"Total documents in this collection: {count}"

    # ===============================
    # TOTAL PAGES
    # ===============================
    if op == "total_pages":
        if notebook_id and not collection_id:
            cur.execute(
                """
                SELECT COALESCE(
                    CASE
                        WHEN dm.document_id IS NOT NULL
                            THEN COALESCE(bm.page_count, dm.page_count, 0)
                        ELSE COALESCE(bm.page_count, 0)
                    END, 0
                )
                FROM (SELECT %s::text AS document_id) seed
                LEFT JOIN document_metadata dm ON dm.document_id = seed.document_id
                LEFT JOIN book_metadata bm ON bm.document_id = seed.document_id
                """,
                (notebook_id,),
            )
            row = cur.fetchone()
            total_pages = row[0] if row else 0
            return f"Total pages in this document: {total_pages}"

        if not collection_id or not user_id:
            return "Metadata scope not available."

        where_sql, params = collection_scope_sql(alias="dm")
        cur.execute(
            f"""
            SELECT COALESCE(
                SUM(
                    COALESCE(
                        CASE
                            WHEN dm.document_role = 'ReferenceBook'
                                THEN COALESCE(bm.page_count, dm.page_count, 0)
                            ELSE COALESCE(dm.page_count, bm.page_count, 0)
                        END, 0
                    )
                ), 0
            )
            FROM document_metadata dm
            LEFT JOIN book_metadata bm ON bm.document_id = dm.document_id
            WHERE {where_sql}
            """,
            params,
        )

        result = cur.fetchone()
        total_pages = result[0] if result and result[0] else 0

        return f"Total pages in this collection: {total_pages}"

    # ===============================
    # LIST DOCUMENTS
    # ===============================
    if op == "list":
        if notebook_id and not collection_id:
            cur.execute(
                """
                SELECT
                    dm.filename,
                    dm.document_type,
                    dm.document_role,
                    dm.page_count,
                    dm.word_count,
                    dm.court_name,
                    dm.order_date,
                    bm.title,
                    bm.section_count,
                    bm.article_count,
                    bm.part_count,
                    bm.schedule_count,
                    bm.chapter_count
                FROM document_metadata dm
                LEFT JOIN book_metadata bm ON bm.document_id = dm.document_id
                WHERE dm.document_id = %s
                """,
                (notebook_id,),
            )
            rows = cur.fetchall()
            if not rows:
                return "No document metadata found."
            return _format_single_doc_row(rows[0])

        if not collection_id or not user_id:
            return "Metadata scope not available."

        where_sql, params = collection_scope_sql(alias="dm")
        cur.execute(
            f"""
            SELECT
                dm.filename,
                dm.document_type,
                dm.document_role,
                dm.court_name,
                dm.order_date,
                bm.title,
                bm.section_count,
                bm.article_count,
                bm.part_count,
                bm.schedule_count,
                bm.chapter_count,
                COALESCE(bm.page_count, dm.page_count, 0) AS page_count
            FROM document_metadata dm
            LEFT JOIN book_metadata bm ON bm.document_id = dm.document_id
            WHERE {where_sql}
            ORDER BY dm.order_date DESC NULLS LAST, dm.created_at DESC NULLS LAST
            """,
            params,
        )

        rows = cur.fetchall()

        if not rows:
            return "No documents found in this collection."

        response = []
        for row in rows:
            filename, doc_type, role, court, date, title, section_count, article_count, part_count, schedule_count, chapter_count, page_count = row
            if _is_book_role(role) or (doc_type or "").lower() == "book":
                response.append(
                    f"- {title or filename} | Book | Pages: {page_count or 0} | "
                    f"Sections: {section_count or 0} | Articles: {article_count or 0} | "
                    f"Parts: {part_count or 0} | Schedules: {schedule_count or 0} | "
                    f"Chapters: {chapter_count or 0}"
                )
            else:
                response.append(
                    f"- {filename} | {doc_type or 'Unknown'} | {court or 'Unknown Court'} | {date or 'No Date'}"
                )

        return "Documents in this collection:\n\n" + "\n".join(response)

    # ===============================
    # LIST DOCUMENTS WITH PAGE COUNTS
    # ===============================
    if op == "list_pages":
        if notebook_id and not collection_id:
            cur.execute(
                """
                SELECT
                    dm.filename,
                    dm.document_type,
                    dm.document_role,
                    COALESCE(bm.title, dm.filename) AS title,
                    COALESCE(bm.page_count, dm.page_count, 0) AS page_count
                FROM document_metadata dm
                LEFT JOIN book_metadata bm ON bm.document_id = dm.document_id
                WHERE dm.document_id = %s
                """,
                (notebook_id,),
            )
            row = cur.fetchone()
            if not row:
                return "No document metadata found."
            filename, doc_type, role, title, page_count = row
            if _is_book_role(role) or (doc_type or "").lower() == "book":
                return f"Document pages:\n\n- {title} | Book | Pages: {page_count or 0}"
            return f"Document pages:\n\n- {filename} | {doc_type or 'Unknown'} | Pages: {page_count or 0}"

        if not collection_id or not user_id:
            return "Metadata scope not available."

        where_sql, params = collection_scope_sql(alias="dm")
        cur.execute(
            f"""
            SELECT
                dm.filename,
                dm.document_type,
                dm.document_role,
                COALESCE(bm.title, dm.filename) AS title,
                COALESCE(bm.page_count, dm.page_count, 0) AS page_count
            FROM document_metadata dm
            LEFT JOIN book_metadata bm ON bm.document_id = dm.document_id
            WHERE {where_sql}
            ORDER BY dm.created_at DESC NULLS LAST
            """,
            params,
        )
        rows = cur.fetchall()
        if not rows:
            return "No documents found in this collection."

        response = []
        for filename, doc_type, role, title, page_count in rows:
            if _is_book_role(role) or (doc_type or "").lower() == "book":
                response.append(f"- {title} | Book | Pages: {page_count or 0}")
            else:
                response.append(f"- {filename} | {doc_type or 'Unknown'} | Pages: {page_count or 0}")

        return "Documents and page counts:\n\n" + "\n".join(response)

    # ===============================
    # LIST JUDGES IN COLLECTION
    # ===============================
    if op in ["filter", "list"] and entities.get("judges"):
        if not collection_id or not user_id:
            return "Metadata scope not available."

        where_sql, params = collection_scope_sql()
        cur.execute(
            f"""
            SELECT filename, judge_name
            FROM document_metadata
            WHERE {where_sql}
              AND document_role = 'Judicial'
            """,
            params,
        )

        rows = cur.fetchall()

        if not rows:
            return "No judge information found in this collection."

        response = []

        for filename, judges in rows:
            if judges:
                response.append(
                    f"- {filename}: {', '.join(judges)}"
                )

        if not response:
            return "Judge information not available."

        return "Judges in this collection:\n\n" + "\n".join(response)




    # ===============================
    # LIST CASES (if explicitly requested)
    # ===============================
    if op == "list" and entities.get("case"):
        if not collection_id or not user_id:
            return "Metadata scope not available."

        where_sql, params = collection_scope_sql()
        cur.execute(
            f"""
            SELECT case_number, document_type, order_date
            FROM document_metadata
            WHERE {where_sql}
              AND document_role = 'Judicial'
            """,
            params,
        )

        rows = cur.fetchall()

        if not rows:
            return "No cases found in this collection."

        return "\n".join(
            f"Case: {r[0] or 'N/A'}, "
            f"Type: {r[1] or 'Unknown'}, "
            f"Order Date: {r[2] or 'N/A'}"
            for r in rows
        )

    return "Metadata intent recognized but not yet supported."
