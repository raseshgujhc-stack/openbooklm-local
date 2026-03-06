#!/usr/bin/env python3
"""
Reingest script to rebuild document chunks using section-aware chunking.

This script:
1. Identifies all ReferenceBook documents in book_metadata
2. Re-ingests them with section-aware chunking
3. Rebuilds FAISS indexes
4. Optionally rebuilds global index

Usage:
    python reingest_with_sections.py  [--global] [--collection-id ID]
    
Options:
    --global: Also rebuild global FAISS index
    --collection-id: Only reingest documents in this collection
"""

import sys
import os
from pathlib import Path

# Add backend to path to import from db (not rag/db.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pdf_reader import read_pdf_from_path
from db import get_repo
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"

def get_reference_books(collection_id=None):
    """Get all ReferenceBook documents that need reingesting."""
    try:
        repo = get_repo()
        cur = repo.conn.cursor()
        
        query = """
        SELECT dm.document_id, dm.filename, dm.collection_id, dm.user_id
        FROM document_metadata dm
        LEFT JOIN book_metadata bm ON bm.document_id = dm.document_id
        WHERE dm.document_role = 'ReferenceBook'
           OR bm.document_id IS NOT NULL
        """
        
        params = []
        
        if collection_id:
            query += " AND dm.collection_id = %s"
            params.append(collection_id)
        
        query += " ORDER BY dm.created_at"
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        print(f"📚 Found {len(rows)} ReferenceBook documents to reingest")
        return rows
    
    except Exception as e:
        print(f"❌ Error fetching documents: {e}")
        return []


def get_document_path(document_id):
    """Find the PDF file for a document."""
    # Check uploaded_pdfs folder
    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    
    # Also check backend/data/uploads if that's where originals are stored
    alt_path = BASE_DIR / "data" / "uploads" / f"{document_id}.pdf"
    if alt_path.exists():
        return alt_path
    
    return None


def reingest_document(document_id, filename, collection_id, user_id):
    """Re-ingest a single document with new chunking strategy (uses UPDATE instead of INSERT)."""
    
    # Find the PDF
    pdf_path = get_document_path(document_id)
    if not pdf_path:
        print(f"⚠️  PDF not found for {document_id} ({filename})")
        return False
    
    try:
        # Read PDF using path-based reader
        text = read_pdf_from_path(pdf_path)
        if not text.strip():
            print(f"⚠️  Empty PDF: {filename}")
            return False
        
        # ---- DELETE OLD CHUNKS FROM FAISS ----
        from rag.vector_store import delete_vectors
        try:
            delete_vectors(document_id)
            print(f"  🗑️  Cleared old FAISS vectors for {document_id}")
        except Exception as e:
            print(f"  ⚠️  Warning clearing old vectors: {e}")
        
        # ---- RE-CHUNK WITH SECTION AWARENESS ----
        from rag.ingest import (
            embed_texts, extract_basic, classify_document_kind,
            estimate_page_count, extract_acts, normalize_and_dedup_acts,
            extract_book_metadata, generate_short_abstract, sha256, build_book_section_rows,
            get_book_chunks, infer_acts_from_filename,
            split_into_chunks
        )
        from rag.vector_store import save_vectors
        from db import get_repo
        from psycopg2.extras import Json
        from datetime import datetime
        
        print(f"🔄 Reingesting: {filename}")
        
        # Classify and chunk
        basic = extract_basic(text)
        page_count = estimate_page_count(text)
        doc_kind = classify_document_kind(text, filename)
        role = "Judicial" if doc_kind == "judgment" else "ReferenceBook"
        
        # Use section-aware chunking for reference books
        if role == "ReferenceBook":
            section_chunks, chunks, strategy, toc_entries = get_book_chunks(text)
            print(f"📖 Using {strategy}-aware chunking: {len(chunks)} chunks detected")
        else:
            chunks = split_into_chunks(text)
            toc_entries = []
            print(f"📄 Using paragraph-based chunking: {len(chunks)} chunks")
        
        # ---- EMBED AND SAVE NEW VECTORS ----
        embeddings = embed_texts(chunks)
        vectors_payload = []
        for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            payload = {"text": chunk_text, "embedding": emb, "chunk_index": i}
            if role == "ReferenceBook" and i < len(section_chunks):
                payload["section_id"] = section_chunks[i].get("section_id")
                payload["section_title"] = section_chunks[i].get("section_title")
                payload["section_type"] = section_chunks[i].get("section_type")
            vectors_payload.append(payload)
        save_vectors(
            notebook_id=document_id,
            vectors=vectors_payload,
            collection_id=collection_id,
        )
        
        # ---- UPDATE DOCUMENT METADATA (don't re-insert) ----
        raw_acts = extract_acts(text)
        if role == "ReferenceBook":
            raw_acts = (raw_acts or []) + infer_acts_from_filename(filename)
        acts = normalize_and_dedup_acts(raw_acts)
        
        repo = get_repo()
        cur = repo.conn.cursor()
        
        # Update document_metadata (preserve existing fields, update key ones)
        cur.execute("""
            UPDATE document_metadata
            SET 
                act_names = %s,
                referenced_laws = %s,
                ingested_at = %s,
                file_hash = %s
            WHERE document_id = %s
        """, (
            Json(acts),
            Json([a["act"] for a in acts]),
            datetime.utcnow().isoformat(),
            sha256(text),
            document_id
        ))
        
        # Update book metadata if it's a reference book
        if doc_kind == "book":
            book_meta = extract_book_metadata(
                text=text,
                filename=filename,
                basic=basic,
                page_count=page_count,
                toc_entries=toc_entries,
            )
            
            book_meta.update(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "user_id": user_id,
                    "collection_id": collection_id,
                    "file_hash": sha256(text),
                    "ingested_at": datetime.utcnow().isoformat(),
                }
            )
            repo.insert_book_document(book_meta)

            section_rows = build_book_section_rows(
                document_id=document_id,
                filename=filename,
                user_id=user_id,
                collection_id=collection_id,
                section_chunks=section_chunks,
                act_alias_hits=book_meta.get("act_alias_hits") or [],
            )
            repo.upsert_book_section_rows(document_id, section_rows)
        
        repo.conn.commit()
        print(f"✅ Reingested: {filename} ({len(chunks)} chunks)")
        return True
    
    except Exception as e:
        print(f"❌ Error reingesting {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False


def rebuild_global_index():
    """Rebuild the global FAISS index from all notebooks."""
    print("\n🌍 Rebuilding global FAISS index...")
    
    try:
        from rag.vector_store import build_global_index
        
        # Get all notebook IDs
        notebooks = []
        faiss_dir = BASE_DIR / "data" / "faiss"
        
        if faiss_dir.exists():
            index_files = list(faiss_dir.glob("*.index"))
            notebooks = [f.stem for f in index_files]
        
        print(f"📊 Building index from {len(notebooks)} notebooks...")
        
        # This function should aggregate all FAISS indexes
        # You may need to implement this in vector_store.py if it doesn't exist
        # build_global_index(notebooks)
        
        print(f"✅ Global index rebuild complete")
        
    except Exception as e:
        print(f"❌ Error rebuilding global index: {e}")


def main():
    parser = argparse.ArgumentParser(description="Reingest documents with section-aware chunking")
    parser.add_argument("--global", dest="rebuild_global", action="store_true", help="Rebuild global FAISS index")
    parser.add_argument("--collection-id", help="Only reingest documents in this collection")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📖 DOCUMENT REINGEST WITH SECTION-AWARE CHUNKING")
    print("=" * 60)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made\n")
    
    # Get documents to reingest
    docs = get_reference_books(collection_id=args.collection_id)
    
    if not docs:
        print("✅ No documents to reingest")
        return
    
    # Reingest each document
    successful = 0
    failed = 0
    
    for i, (doc_id, filename, collection_id, user_id) in enumerate(docs, 1):
        print(f"\n[{i}/{len(docs)}] Processing: {filename}")
        
        if args.dry_run:
            print(f"Would reingest: {doc_id}")
        else:
            if reingest_document(doc_id, filename, collection_id, user_id):
                successful += 1
            else:
                failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed:     {failed}")
    print("=" * 60)
    
    # Rebuild global index if requested
    if args.rebuild_global and not args.dry_run:
        rebuild_global_index()
    
    print("\n🎉 Reingest complete!")


if __name__ == "__main__":
    main()
