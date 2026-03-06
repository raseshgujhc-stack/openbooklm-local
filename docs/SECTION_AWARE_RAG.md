# Section-Aware Legal Document RAG — Architecture

## Problem Solved

**Before:** Mixing act contents where sections exist in multiple acts
- Query: "What is Section 151?"
- ❌ Result: Mixed answers combining Code of Civil Procedure (CPC) + Bharatiya Nyaya Sanhita (BNS)
- Root cause: Random paragraph chunking + semantic search returned all similar chunks

**After:** Precise section-level retrieval
- Query: "What is Section 151 of the Code of Civil Procedure?"
- ✅ Result: **CPC Section 151** with act name clearly labeled
- Query: "Section 151 of BNS"
- ✅ Result: **BNS Section 151** — not mixed with CPC

---

## Architecture

### 1. **Section-Aware Chunking** (`chunker.py`)

Instead of paragraph-based chunking:
```
Old approach:
  Chunk 1: "...paragraph text... Section 151 discusses... more text..."
  Chunk 2: "...other paragraph..."

New approach:
  Chunk 1: "Section 151 – [FULL section 151 text]"
  Chunk 2: "Section 152 – [FULL section 152 text]"
  Chunk 3: "Section 153 – [FULL section 153 text]"
```

**Functions:**
- `chunk_by_sections(text, doc_type="book")` — Regex-based section detection
  - Detects: `Section 123`, `Article 456`, `Chapter 789`, `Part II`, etc.
  - Returns: List of `{section_id, section_title, text, full_text, has_section_marker}`
  - Fallback: Uses paragraph chunking if no sections found

- `extract_section_from_question(question)` — Parse user questions
  - Input: "What is Section 151 of the Code of Civil Procedure?"
  - Output: `("151", "section")`

### 2. **Act Metadata Query** (`section_retrieval.py`)

Leverages existing `book_metadata` table:

```sql
SELECT COALESCE(act_alias_hits, '[]'::jsonb), section_count
FROM book_metadata
WHERE document_id = %s
```

**Key functions:**
- `find_act_for_document(document_id)` — Get acts from `act_alias_hits` JSONB field
- `find_documents_with_act(collection_id, user_id, act_name)` — Query `book_metadata` to find docs containing the requested act
- `find_section_in_notebook(notebook_id, section_num)` — Direct section lookup in chunk metadata

### 3. **Section-First Retrieval Strategy** (`section_retrieval.py`)

When user asks about a section:

```
Step 1: Extract section number → "151"
        ↓
Step 2: Query book_metadata for matching acts
        ↓
Step 3: Search for exact section match in FAISS metadata
        ✅ Found? → Return immediately (perfect precision)
        ❌ Not found? → Fallback to semantic search
```

**Function:** `retrieve_by_section_first()`
- Returns: `(chunks, strategy)` where strategy is:
  - `"section_exact_match"` — Found exact section
  - `"section_not_found_fallback_to_semantic"` — No exact match, use embeddings

### 4. **Integration into RAG Pipeline** (`rag_pipeline.py`)

New flow in `generate_answer()`:

```
0️⃣  Metadata routing (unchanged)
    ↓
0.75️⃣ SECTION-FIRST RETRIEVAL ← NEW STEP
    └─ If found exact section → Return immediately
    \
0.5️⃣ Generic metadata QA (unchanged)
    ↓
1️⃣  Router + embedding (unchanged)
    ↓
2️⃣  Semantic search (fallback if section-first fails)
```

---

## How Act Metadata is Used

### book_metadata Table Structure

```sql
CREATE TABLE book_metadata (
  document_id      TEXT PRIMARY KEY,
  act_alias_hits   JSONB,        -- ["Code of Civil Procedure", "CPC", ...] 
  section_count    INTEGER,      -- 500+ for huge acts
  chapter_titles   JSONB,        -- ["CHAPTER 1: ...", "CHAPTER 2: ..."]
  inferred_subjects JSONB,       -- ["Criminal Law", "Procedure", ...]
  ...
);
```

### Retrieval Process

1. **User asks:** "What is Section 151?"
   - Extract: section_num = "151", section_type = "section"

2. **Query book_metadata:**
   ```sql
   SELECT document_id FROM book_metadata
   WHERE act_alias_hits::text ILIKE '%code of civil%'
      OR act_alias_hits::text ILIKE '%cpc%'
   ```

3. **Search notebooks for exact match:**
   - Load FAISS indexes for found documents
   - Scan chunk metadata for `Section 151`
   - Return matching chunk immediately

4. **Label result with act name:**
   - Fetch `act_alias_hits` from book_metadata
   - Include in LLM context: `[Source: Code of Civil Procedure]`

---

## Usage & Reingest

### First Time Setup

1. **Ensure legal acts are ingested with new strategy:**

```bash
cd backend
python -m rag.reingest_with_sections
```

This will:
- Identify all `ReferenceBook` documents
- Re-chunk using section-aware strategy
- Rebuild FAISS indexes with section metadata

2. **Optionally rebuild global index:**

```bash
python -m rag.reingest_with_sections --global
```

### Query Examples

**Query 1: Specific act**
```
User: "What is Section 151 of the Code of Civil Procedure?"
System: 
  1. Extract: Section 151, Code of Civil Procedure
  2. Query book_metadata → Find CPC document
  3. Find exact "Section 151" in CPC chunks
  4. Result: CPC Section 151 (NOT BNS Section 151)
  Mode: section_exact ✅
```

**Query 2: Ambiguous section**
```
User: "What is Section 151?"
System:
  1. Extract: Section 151, (no specific act mentioned)
  2. Search across all documents
  3. Find multiple matches (CPC + BNS + IPC)
  4. Fallback: Use semantic search to disambiguate
  Mode: section_fallback_semantic
```

**Query 3: Non-section query**
```
User: "What are fundamental rights?"
System:
  1. No section detected
  2. Use semantic search normally
  3. Return relevant chunks from Constitution, etc.
  Mode: semantic
```

---

## Implementation Details

### New Files

- **[chunker.py](chunker.py)** — `chunk_by_sections()`, `extract_section_from_question()`
- **[section_retrieval.py](section_retrieval.py)** — Complete section retrieval logic
- **[reingest_with_sections.py](reingest_with_sections.py)** — Migration script
- **[reingest_with_sections.sh](reingest_with_sections.sh)** — Shell wrapper

### Modified Files

- **[ingest.py](ingest.py)** — Use section-aware chunking for ReferenceBooks
- **[rag_pipeline.py](rag_pipeline.py)** — Add section-first retrieval (Step 0.75)

### Backward Compatibility

✅ **Fully compatible:**
- Judgments/cases still use paragraph chunking
- Non-section queries use semantic search
- Global index, metadata queries unaffected

---

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| Section precision | Mixed from multiple acts | Exact match per act |
| Query speed | All semantic search | Fast regex lookup first |
| Chunk quality | Random paragraph boundaries | Respects section structure |
| Act distinction | Blended together | Clear metadata labeling |
| Metadata usage | Unused act_alias_hits | Leveraged via queries |
| Reingest needed | N/A | One-time migration |

---

## Next Steps

1. **Run reingest:**
   ```bash
   python backend/rag/reingest_with_sections.py --global
   ```

2. **Test queries:**
   ```
   - "What is Section 151 of the Code of Civil Procedure?"
   - "Section 12 of the Indian Constitution"
   - "Article 21 - right to life"
   ```

3. **Monitor** the response mode in API output:
   - `"mode": "section_exact"` → Perfect match ✅
   - `"mode": "section_fallback_semantic"` → Used embeddings as fallback
   - `"mode": "semantic"` → No section detected, normal RAG

---

## Troubleshooting

**Q: Sections not being detected?**
- Check section pattern in `chunk_by_sections()` regex
- Add custom patterns for non-standard formats
- Fallback to paragraph chunking automatically applied

**Q: Act metadata not populating?**
- Ensure `book_metadata.act_alias_hits` is populated during ingest
- Check `ingest.py:extract_book_metadata()` function
- Re-run book metadata extraction if needed

**Q: Performance concerns?**
- Section regex matching is O(1) per document
- FAISS lookups are O(log n)  
- Fallback to semantic search for truly ambiguous queries
- Global index acceleration possible with HNSW tuning

---

For questions, refer to the section retrieval test files or inline code comments.
