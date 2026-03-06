"""
Section-aware retrieval strategy for legal documents.

Flow:
1. Extract section number(s) from question via regex
2. Query book_metadata to find documents with matching act
3. Perform exact section lookup in those notebooks
4. Fallback to semantic search if exact section not found
"""

import re
import sys
from pathlib import Path
import numpy as np
from typing import Optional, List, Dict

# Add backend to path to import from db (not rag/db.py)
sys.path.insert(0, str(Path(__file__).parent.parent))
from db import get_repo
from rag.chunker import extract_section_from_question, extract_all_sections
from rag.vector_store import load_vectors
from rag.act_catalog import get_alias_map, normalize_act_name

SECTION_RETRIEVAL_REV = "section_retrieval_rev_2026_03_01_01"


def _is_global_collection(collection_id: str) -> bool:
    """
    Check whether a collection is globally visible.
    """
    if not collection_id:
        return False
    try:
        repo = get_repo()
        cur = repo.conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(is_global, FALSE)
            FROM collections
            WHERE collection_id = %s
            """,
            (collection_id,),
        )
        row = cur.fetchone()
        return bool(row[0]) if row else False
    except Exception:
        return False


def find_act_for_document(document_id: str) -> List[str]:
    """
    Fetch act aliases for a document from book_metadata.
    
    Returns: List of act names from act_alias_hits JSONB field
    """
    try:
        repo = get_repo()
        cur = repo.conn.cursor()
        
        cur.execute(
            """
            SELECT COALESCE(act_alias_hits, '[]'::jsonb)
            FROM book_metadata
            WHERE document_id = %s
            """,
            (document_id,),
        )
        
        row = cur.fetchone()
        if row:
            acts = row[0]
            return acts if isinstance(acts, list) else []
        return []
    except Exception as e:
        print(f"❌ Error fetching act aliases: {e}")
        return []


def find_documents_with_act(collection_id: str = None, user_id: str = None, act_name: str = None) -> List[str]:
    """
    Query book_metadata to find all documents containing a specific act.
    
    Returns: List of document_ids
    """
    if not act_name:
        return []
    
    try:
        repo = get_repo()
        cur = repo.conn.cursor()
        
        query = """
        SELECT document_id
        FROM book_metadata
        WHERE EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(COALESCE(act_alias_hits, '[]'::jsonb)) elem
            WHERE LOWER(elem) = LOWER(%s)
        )
        """
        params = [act_name]
        
        # Add collection/user filters if provided.
        # Remark: for global collections, scope by collection only so admins/users
        # can query all globally shared book docs.
        if collection_id and user_id:
            if _is_global_collection(collection_id):
                query += " AND collection_id = %s"
                params = [act_name, collection_id]
            else:
                query += " AND collection_id = %s AND user_id = %s"
                params = [act_name, collection_id, user_id]
        elif collection_id:
            query += " AND collection_id = %s"
            params = [act_name, collection_id]
        
        cur.execute(query, params)
        rows = cur.fetchall()
        return [row[0] for row in rows if row[0]]
    
    except Exception as e:
        print(f"❌ Error querying documents by act: {e}")
        return []


def extract_book_phrase_from_question(question: str) -> Optional[str]:
    """
    Extract book/act phrase from patterns like:
    - "Section 33 of Negotiable Instrument Act"
    - "Section 33 in BNSS Act"
    - "Section 33 under CPC"
    """
    q = question or ""
    m = re.search(
        r"\b(?:section|article|chapter|part|schedule)\s+[0-9A-Za-z\(\)]+\s+(?:of|in|under)\s+(?:the\s+)?([A-Za-z0-9_][A-Za-z0-9_\s\.\,&()\-]{2,220})",
        q,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    phrase = re.sub(r"\s+", " ", m.group(1)).strip(" .,:;-")
    # Trim common trailing conversational tails from follow-up queries.
    phrase = re.sub(
        r"\s+\b(?:what|which|who|how|tell|explain|meaning|describe)\b.*$",
        "",
        phrase,
        flags=re.IGNORECASE,
    ).strip(" .,:;-")
    return phrase or None


def _normalize_book_key(value: str) -> str:
    s = (value or "").lower()
    s = s.replace(".pdf", " ")
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_tokens_for_match(value: str) -> List[str]:
    base = _normalize_book_key(value)
    toks = [t for t in base.split() if t]
    stop = {"the", "of", "and", "an", "a"}
    out = []
    for t in toks:
        if t in stop:
            continue
        # Simple singularization helps "instrument" ~= "instruments".
        if len(t) > 4 and t.endswith("s"):
            t = t[:-1]
        out.append(t)
    return out


def _book_acronyms(value: str) -> set[str]:
    toks = _normalize_tokens_for_match(value)
    if len(toks) < 2:
        return set()
    tail_words = {"act", "code", "sanhita", "adhiniyam", "rule", "rules", "regulation", "regulations", "law"}
    ignore_short = tail_words.union({"book", "section", "article", "chapter", "part", "schedule"})
    out = set()
    alpha_toks = [t for t in toks if t.isalpha()]
    # De-duplicate while preserving order (filename + title often repeat same words).
    alpha_toks = list(dict.fromkeys(alpha_toks))
    # Preserve explicit short-form tokens like "ni", "bnss", "crpc".
    for t in alpha_toks:
        if 2 <= len(t) <= 8 and t not in ignore_short:
            out.add(t)
    full = "".join(t[0] for t in alpha_toks if t)
    if len(full) >= 2:
        out.add(full)
    if alpha_toks and alpha_toks[-1] in tail_words and len(alpha_toks) >= 3:
        stem = alpha_toks[:-1]
        stem_acr = "".join(t[0] for t in stem if t and t[0].isalnum())
        if len(stem_acr) >= 2:
            out.add(stem_acr)
    return out


def find_documents_by_book_phrase(
    phrase: str,
    collection_id: str = None,
    user_id: str = None,
) -> List[str]:
    """
    Fallback resolver when act alias map doesn't recognize the phrase.
    Matches by filename/title of reference books.
    """
    if not phrase:
        return []
    try:
        repo = get_repo()
        cur = repo.conn.cursor()
        query = """
            SELECT DISTINCT dm.document_id, dm.filename, bm.title
            FROM document_metadata dm
            LEFT JOIN book_metadata bm ON bm.document_id = dm.document_id
            WHERE dm.document_role = 'ReferenceBook'
        """
        params = []
        if collection_id and user_id:
            if _is_global_collection(collection_id):
                query += " AND dm.collection_id = %s"
                params.append(collection_id)
            else:
                query += " AND dm.collection_id = %s AND dm.user_id = %s"
                params.extend([collection_id, user_id])
        elif collection_id:
            query += " AND dm.collection_id = %s"
            params.append(collection_id)
        elif user_id:
            query += " AND dm.user_id = %s"
            params.append(user_id)
        cur.execute(query, params)
        rows = cur.fetchall()

        norm_phrase = _normalize_book_key(phrase)
        if not norm_phrase:
            return []
        phrase_tokens = _normalize_tokens_for_match(norm_phrase)
        phrase_token_set = set(phrase_tokens)
        phrase_acronyms = _book_acronyms(norm_phrase)

        matched = []
        exact_matched = []
        for doc_id, filename, title in rows:
            norm_filename = _normalize_book_key(filename or "")
            norm_title = _normalize_book_key(title or "")
            cand = _normalize_book_key(f"{filename or ''} {title or ''}")
            if not cand:
                continue
            cand_tokens = set(_normalize_tokens_for_match(cand))
            cand_acronyms = _book_acronyms(cand)
            # Strict filename/title equality first (most reliable for UI-provided names).
            if norm_phrase == norm_filename or norm_phrase == norm_title:
                exact_matched.append(doc_id)
                continue
            if norm_phrase in cand or cand in norm_phrase:
                matched.append(doc_id)
                continue
            if phrase_token_set and phrase_token_set.issubset(cand_tokens):
                matched.append(doc_id)
                continue
            if phrase_acronyms and cand_acronyms and phrase_acronyms.intersection(cand_acronyms):
                matched.append(doc_id)
        if exact_matched:
            return list(dict.fromkeys(exact_matched))
        return list(dict.fromkeys(matched))
    except Exception as e:
        print(f"❌ Error finding docs by phrase: {e}")
        return []


def extract_acts_from_question(question: str) -> List[str]:
    """
    Extract canonical act mentions from user question using catalog aliases.
    """
    text = question or ""
    lowered = text.lower()
    alias_map = get_alias_map()
    if not alias_map:
        return []

    hits = []
    aliases = sorted(alias_map.keys(), key=len, reverse=True)
    for alias in aliases:
        if len(alias) < 4:
            continue
        parts = [p for p in alias.split(" ") if p]
        if not parts:
            continue
        separator = r"[\s\.\-()/,]*"
        pattern = rf"(?<!\w){separator.join(re.escape(p) for p in parts)}(?!\w)"
        m = re.search(pattern, lowered)
        if not m:
            continue
        hits.append((m.start(), alias_map[alias]))

    hits.sort(key=lambda x: x[0])
    ordered = []
    seen = set()
    for _, canon in hits:
        if canon in seen:
            continue
        seen.add(canon)
        ordered.append(canon)
    # Fallback path for short legal abbreviations that may be absent in alias seed
    # (e.g. "NI Act", "MV Act", "Evidence Act").
    fallback_patterns = [
        r"\b([A-Za-z]{1,8}\s+Act)\b",
        r"\b([A-Za-z][A-Za-z\s]{2,80}\s+Act)\b",
        r"\b([A-Za-z][A-Za-z\s]{2,80}\s+Code)\b",
        r"\b([A-Za-z][A-Za-z\s]{2,80}\s+Sanhita)\b",
        r"\b([A-Za-z][A-Za-z\s]{2,80}\s+Adhiniyam)\b",
    ]
    for pat in fallback_patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            candidate = re.sub(r"\s+", " ", m.group(1)).strip()
            canon = normalize_act_name(candidate)
            if canon and canon not in seen:
                seen.add(canon)
                ordered.append(canon)

    return ordered


def find_section_in_notebook(notebook_id: str, section_num: str) -> Optional[Dict]:
    """
    Look for exact section match in a notebook's chunks.
    
    Returns: First chunk matching the section, or None
    """
    try:
        loaded = load_vectors(notebook_id)
        if not loaded:
            return None
        
        index, metadata = loaded
        
        # Try multiple section marker patterns:
        # 1. "Section 151. " (named format)
        # 2. "151. " (numbered format) 
        # 3. "Article 12" (constitution format)
        patterns = [
            rf'\b(?:Section|Article|Chapter|Part)\s+{re.escape(section_num)}[\.\s\(]',  # "Section 151. " or "Article 12"
            rf'^{re.escape(section_num)}\.\s+',  # "151. " at start of text
            rf'\n{re.escape(section_num)}\.\s+',  # "151. " after newline
        ]
        
        for idx, chunk_meta in enumerate(metadata):
            chunk_text = chunk_meta.get("text", "")
            
            # Try any of the patterns
            for pattern in patterns:
                if re.search(pattern, chunk_text, re.MULTILINE):
                    return {
                        "notebook_id": notebook_id,
                        "chunk_index": idx,
                        "text": chunk_meta.get("text"),
                        "section_id": section_num,
                        "match_type": "exact",
                        "act_names": chunk_meta.get("act_names", []),
                    }
        
        return None
    
    except Exception as e:
        print(f"❌ Error finding section in notebook: {e}")
        return None


def find_sections_in_index(
    sections: List[tuple[str, str]],
    act_names: List[str],
    collection_id: str = None,
    user_id: str = None,
    notebook_id: str = None,
) -> List[Dict]:
    """
    Exact Act+Section lookup from normalized index table.
    """
    if not sections or not act_names:
        return []
    try:
        repo = get_repo()
        cur = repo.conn.cursor()

        section_codes = [num.upper() for num, _ in sections if num]
        if not section_codes:
            return []

        query = """
            SELECT document_id, chunk_index, section_code, act_canonical, text_preview
            FROM book_section_index
            WHERE act_canonical = ANY(%s)
              AND section_code = ANY(%s)
        """
        params = [act_names, section_codes]
        if notebook_id:
            query += " AND document_id = %s"
            params.append(notebook_id)
        elif collection_id and user_id:
            if _is_global_collection(collection_id):
                query += " AND collection_id = %s"
                params.append(collection_id)
            else:
                query += " AND collection_id = %s AND user_id = %s"
                params.extend([collection_id, user_id])
        elif collection_id:
            query += " AND collection_id = %s"
            params.append(collection_id)
        elif user_id:
            query += " AND user_id = %s"
            params.append(user_id)

        query += " ORDER BY document_id, chunk_index LIMIT 60"
        cur.execute(query, params)
        rows = cur.fetchall()
        out = []
        for doc_id, chunk_index, section_code, act_canonical, text_preview in rows:
            out.append(
                {
                    "notebook_id": doc_id,
                    "chunk_index": chunk_index,
                    "section_id": section_code,
                    "match_type": "exact",
                    "act_names": [act_canonical],
                    "preview": text_preview or "",
                }
            )
        return out
    except Exception as e:
        print(f"❌ Error querying book_section_index: {e}")
        return []


def find_sections_without_act(
    sections: List[tuple[str, str]],
    collection_id: str = None,
    user_id: str = None,
    notebook_id: str = None,
) -> List[Dict]:
    """
    Section-only lookup used to detect ambiguity when Act is not mentioned.
    """
    if not sections:
        return []
    try:
        repo = get_repo()
        cur = repo.conn.cursor()
        section_codes = [num.upper() for num, _ in sections if num]
        if not section_codes:
            return []
        query = """
            SELECT document_id, chunk_index, section_code, act_canonical, text_preview
            FROM book_section_index
            WHERE section_code = ANY(%s)
        """
        params = [section_codes]
        if notebook_id:
            query += " AND document_id = %s"
            params.append(notebook_id)
        elif collection_id and user_id:
            if _is_global_collection(collection_id):
                query += " AND collection_id = %s"
                params.append(collection_id)
            else:
                query += " AND collection_id = %s AND user_id = %s"
                params.extend([collection_id, user_id])
        elif collection_id:
            query += " AND collection_id = %s"
            params.append(collection_id)
        elif user_id:
            query += " AND user_id = %s"
            params.append(user_id)

        query += " ORDER BY act_canonical, document_id, chunk_index LIMIT 100"
        cur.execute(query, params)
        rows = cur.fetchall()
        return [
            {
                "notebook_id": doc_id,
                "chunk_index": chunk_index,
                "section_id": section_code,
                "match_type": "exact",
                "act_names": [act_canonical],
                "preview": text_preview or "",
            }
            for doc_id, chunk_index, section_code, act_canonical, text_preview in rows
        ]
    except Exception as e:
        print(f"❌ Error querying section-only index: {e}")
        return []


def _attach_notebook_filenames(hits: List[Dict]) -> List[Dict]:
    if not hits:
        return hits
    try:
        repo = get_repo()
        cur = repo.conn.cursor()
        notebook_ids = sorted({h.get("notebook_id") for h in hits if h.get("notebook_id")})
        if not notebook_ids:
            return hits
        placeholders = ",".join(["%s"] * len(notebook_ids))
        cur.execute(
            f"""
            SELECT document_id, filename
            FROM document_metadata
            WHERE document_id IN ({placeholders})
            """,
            tuple(notebook_ids),
        )
        name_map = {row[0]: row[1] for row in cur.fetchall()}
        out = []
        for h in hits:
            row = dict(h)
            nb = row.get("notebook_id")
            if nb and name_map.get(nb):
                row["filename"] = name_map[nb]
            out.append(row)
        return out
    except Exception:
        return hits


def retrieve_by_section_first(
    question: str,
    notebook_id: str = None,
    collection_id: str = None,
    user_id: str = None,
) -> tuple[List[Dict], str]:
    """
    Section-first retrieval strategy:
    
    1. Extract section number(s) from question (handles "Section 151 and 152")
    2. Find documents with matching act (from book_metadata)
    3. Search for exact section matches in those documents
    4. Return matches + strategy used
    
    Returns: (list of chunks, strategy_used)
    """
    
    # =========================================
    # Step 1: Extract ALL sections from question
    # =========================================
    print(f"🧪 Section retrieval revision: {SECTION_RETRIEVAL_REV}")
    sections = extract_all_sections(question)
    
    if not sections:
        return [], "no_section_detected"
    
    section_info = ", ".join([f"{t.upper()} {n}" for n, t in sections])
    print(f"🔍 Extracted: {section_info} from question")
    
    # =========================================
    # Step 2: Find notebook(s) to search
    # =========================================
    
    notebooks_to_search = []
    
    if notebook_id:
        notebooks_to_search = [notebook_id]
    elif collection_id or user_id:
        # Get all notebooks in collection/user
        try:
            repo = get_repo()
            cur = repo.conn.cursor()
            
            if collection_id and user_id:
                if _is_global_collection(collection_id):
                    # Global collections should include all shared docs in scope.
                    cur.execute(
                        """
                        SELECT DISTINCT document_id
                        FROM document_metadata
                        WHERE collection_id = %s
                          AND document_role = 'ReferenceBook'
                        """,
                        (collection_id,),
                    )
                else:
                    # Private/user collections: keep strict ownership scoping.
                    cur.execute(
                        """
                        SELECT DISTINCT document_id
                        FROM document_metadata
                        WHERE collection_id = %s AND user_id = %s
                          AND document_role = 'ReferenceBook'
                        """,
                        (collection_id, user_id),
                    )
            elif collection_id:
                # Just collection (for reference books with user_id=None)
                cur.execute(
                    """
                    SELECT DISTINCT document_id
                    FROM document_metadata
                    WHERE collection_id = %s
                      AND document_role = 'ReferenceBook'
                    """,
                    (collection_id,),
                )
            else:
                # Just user
                cur.execute(
                    """
                    SELECT DISTINCT document_id
                    FROM document_metadata
                    WHERE user_id = %s
                      AND document_role = 'ReferenceBook'
                    """,
                    (user_id,),
                )
            notebooks_to_search = [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"❌ Error fetching collection notebooks: {e}")
            notebooks_to_search = []
    
    if not notebooks_to_search:
        return [], "no_notebooks_found"

    # Try to constrain by canonical act mentions in question.
    act_mentions = extract_acts_from_question(question)
    if act_mentions:
        print(f"🔎 Act mentions extracted: {act_mentions}")
    else:
        print("🔎 Act mentions extracted: none")

    # Explicit phrase resolver for queries like:
    # "Section 38 of NI Act" / "Section 38 of Negotiable Instrument Act"
    phrase = extract_book_phrase_from_question(question)
    if phrase:
        print(f"🔎 Book phrase extracted: '{phrase}'")
    else:
        print("🔎 Book phrase extracted: none")

    explicit_phrase_scoped = False

    # If alias extraction missed, try normalizing the explicit phrase as an act.
    if not act_mentions and phrase:
        phrase_act = normalize_act_name(phrase)
        if phrase_act:
            act_mentions = [phrase_act]
            print(f"🎯 Phrase normalized to Act: {phrase_act}")

    # If still no Act mention, scope by explicit book/title phrase before ambiguity checks.
    if not act_mentions and phrase:
        phrase_docs = set(
            find_documents_by_book_phrase(
                phrase=phrase,
                collection_id=collection_id,
                user_id=user_id,
            )
        )
        if phrase_docs:
            notebooks_to_search = [nb for nb in notebooks_to_search if nb in phrase_docs]
            print(f"🎯 Book-phrase scoped retrieval: '{phrase}' -> {len(notebooks_to_search)} notebook(s)")
            explicit_phrase_scoped = True
            if not notebooks_to_search:
                return [], "section_book_not_found_in_scope"
        else:
            # User specified a book/act phrase but it does not exist in current scope.
            return [], "section_book_not_found_in_scope"

    if act_mentions:
        indexed_hits = find_sections_in_index(
            sections=sections,
            act_names=act_mentions,
            collection_id=collection_id,
            user_id=user_id,
            notebook_id=notebook_id,
        )
        if indexed_hits:
            hydrated_hits = []
            for hit in indexed_hits:
                loaded = load_vectors(hit["notebook_id"])
                if not loaded:
                    continue
                _, meta = loaded
                idx = int(hit["chunk_index"])
                if idx < 0 or idx >= len(meta):
                    continue
                row = dict(hit)
                row["text"] = meta[idx].get("text", row.get("preview", ""))
                hydrated_hits.append(row)
            if hydrated_hits:
                print(f"🎯 Index match: {len(hydrated_hits)} Act+Section chunk(s)")
                return hydrated_hits, "section_exact_match"
    else:
        if explicit_phrase_scoped:
            # User already specified an explicit book/act phrase and we narrowed scope.
            # Do not re-run collection-wide ambiguity checks.
            section_only_hits = []
        else:
        # If Act is not mentioned, detect ambiguity for same section across Acts.
            section_only_hits = find_sections_without_act(
                sections=sections,
                collection_id=collection_id,
                user_id=user_id,
                notebook_id=notebook_id,
            )
        if section_only_hits:
            acts_found = sorted({(h.get("act_names") or [""])[0] for h in section_only_hits if h.get("act_names")})
            notebook_found = sorted({h.get("notebook_id") for h in section_only_hits if h.get("notebook_id")})
            if len(notebook_found) > 1:
                print(f"⚠️  Ambiguous section across books: notebooks={len(notebook_found)}")
                return _attach_notebook_filenames(section_only_hits[:24]), "section_ambiguous_missing_book"
            if len(acts_found) > 1:
                print(f"⚠️  Ambiguous section without Act: candidates={acts_found[:6]}")
                return [], "section_ambiguous_missing_act"
            hydrated_hits = []
            for hit in section_only_hits:
                loaded = load_vectors(hit["notebook_id"])
                if not loaded:
                    continue
                _, meta = loaded
                idx = int(hit["chunk_index"])
                if idx < 0 or idx >= len(meta):
                    continue
                row = dict(hit)
                row["text"] = meta[idx].get("text", row.get("preview", ""))
                hydrated_hits.append(row)
            if hydrated_hits:
                print(f"🎯 Section-only exact match: {len(hydrated_hits)} chunk(s)")
                return hydrated_hits, "section_exact_match"

    if act_mentions:
        act_scoped_docs = set()
        for act in act_mentions:
            act_scoped_docs.update(
                find_documents_with_act(
                    collection_id=collection_id,
                    user_id=user_id,
                    act_name=act,
                )
            )
        if act_scoped_docs:
            notebooks_to_search = [nb for nb in notebooks_to_search if nb in act_scoped_docs]
            print(f"🎯 Act-scoped retrieval: {act_mentions} -> {len(notebooks_to_search)} notebook(s)")
            if not notebooks_to_search:
                return [], "section_act_not_found_in_scope"
        else:
            print(f"⚠️  Act mentioned but no matching docs in scope: {act_mentions}")
            # Fail-safe: do not broaden to all notebooks when user specified an Act.
            return [], "section_act_not_found_in_scope"
    else:
        # No act/book phrase provided; proceed with current notebook scope.
        pass

    print(f"📚 Searching {len(notebooks_to_search)} notebook(s)")
    
    # =========================================
    # Step 3: Search for all mentioned sections
    # =========================================
    
    exact_matches = []
    
    for section_num, section_type in sections:
        for nb_id in notebooks_to_search:
            match = find_section_in_notebook(nb_id, section_num)
            if match:
                exact_matches.append(match)
                print(f"✅ Found {section_type.upper()} {section_num} in {nb_id[:8]}...")
    
    if exact_matches:
        print("🧭 Section strategy: section_exact_match")
        return exact_matches, "section_exact_match"
    
    print(f"⚠️  No exact section matches found, falling back to semantic search")
    print("🧭 Section strategy: section_not_found_fallback_to_semantic")
    return [], "section_not_found_fallback_to_semantic"


def hybrid_section_semantic_retrieval(
    question: str,
    embedding_query,
    notebook_id: str = None,
    collection_id: str = None,
    user_id: str = None,
    top_k: int = 5,
) -> tuple[List[Dict], str]:
    """
    Hybrid retrieval: Try section-first, then fallback to semantic search.
    
    Returns: (scored_chunks, strategy_used)
    """
    
    # Try section-first retrieval
    section_chunks, strategy = retrieve_by_section_first(
        question,
        notebook_id=notebook_id,
        collection_id=collection_id,
        user_id=user_id,
    )
    
    if section_chunks:
        # Convert to scored format
        scored = []
        for chunk in section_chunks:
            scored.append({
                "distance": 0.0,  # Perfect match
                "score": 1.0,
                "text": chunk.get("text"),
                "notebook_id": chunk.get("notebook_id"),
                "chunk_index": chunk.get("chunk_index"),
                "act_names": chunk.get("act_names", []),
                "section_id": chunk.get("section_id"),
                "match_type": "exact",
                "source_type": "section_exact",
            })
        return scored, f"section_exact_match"
    
    # Fallback: Semantic search
    print(f"🔄 Falling back to semantic search")
    return [], f"section_fallback_semantic"
