# rag/chunker.py
import re

def chunk_text(text, chunk_size=800, overlap=150):
    """
    Semantic-safe chunking for legal documents.
    """

    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]

    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) <= chunk_size:
            current += " " + p
        else:
            chunks.append(current.strip())
            current = current[-overlap:] + " " + p

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_by_sections(text, doc_type: str = "book"):
    """
    Section-aware chunking for legal documents (acts, statutes, etc).
    
    Instead of random paragraph chunking, identify SECTION markers and
    create chunks at section boundaries.
    
    Supports multiple formats:
    - Named format: "Section 123 – text"
    - Named format: "Section 123(1) – text"  
    - Numbered format: "151. text" (common in CPC, IPC)
    - Named format: "Article 456 – text"
    - Named format: "Chapter 789 – text"
    
    Returns: List of dicts with {section_id, section_title, text, full_text}
    """
    
    # Try BOTH named and numbered formats, use whichever produces more matches
    # This prevents edge cases where a stray match triggers the wrong path
    
    named_pattern = r'(?:^|\n)((?:Section|Article|Chapter|Part|Schedule|Clause|Para(?:graph)?)\s+(?:[0-9A-Za-z\-\(\)\.]*?))\s*(?:–|:|—)\s*(.*?)(?=(?:^|\n)(?:Section|Article|Chapter|Part|Schedule|Clause|Para(?:graph)?)\s+[0-9A-Za-z]|\Z)'
    matches = list(re.finditer(named_pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE))
    
    numbered_pattern = r'(?:^|\n)(\d+[A-Za-z]*)\.\s+([^\n]*(?:\n(?!\d+[A-Za-z]*\.\s).*)*)'
    numbered_matches = list(re.finditer(numbered_pattern, text, re.MULTILINE))
    
    # Use whichever pattern yielded more results (likely the correct format)
    # This handles cases where CPC has 2600+ numbered sections vs 1-2 stray named patterns
    if numbered_matches and (not matches or len(numbered_matches) > len(matches)):
        # Using numbered format (Section 151., 152., etc)
        chunks = []
        for match in numbered_matches:
            section_num = match.group(1)
            section_content = match.group(2).strip()
            
            full_text = f"{section_num}. {section_content}"
            chunks.append({
                "section_id": section_num,
                "section_type": "section",
                "section_title": f"Section {section_num}",
                "text": section_content[:2000],
                "full_text": full_text,
                "has_section_marker": True,
            })
        
        return chunks
    
    # Handle named format matches
    if matches:
        chunks = []
        for match in matches:
            section_header = match.group(1).strip()
            section_content = match.group(2).strip()
            
            # Extract section number for easier filtering
            section_num_match = re.search(r'(\d+(?:[A-Za-z\-]*)?)', section_header)
            section_num = section_num_match.group(1) if section_num_match else None
            
            # Extract section type (Section, Article, Chapter, etc)
            section_type_match = re.match(r'(Section|Article|Chapter|Part|Schedule|Clause|Para(?:graph)?)', section_header, re.IGNORECASE)
            section_type = section_type_match.group(1).lower() if section_type_match else "section"
            
            # Create chunk
            full_text = f"{section_header} – {section_content}"
            
            chunks.append({
                "section_id": section_num,
                "section_type": section_type,
                "section_title": section_header,
                "text": section_content[:2000],  # Limit individual section to 2000 chars for embedding
                "full_text": full_text,
                "has_section_marker": True,
            })
        
        return chunks
    
    # Fallback to paragraph chunking if no sections detected
    return [
        {
            "section_id": None,
            "section_title": None,
            "text": chunk,
            "full_text": chunk,
            "has_section_marker": False,
        }
        for chunk in chunk_text(text)
    ]


def extract_toc_entries(text: str, max_entries: int = 200) -> list[str]:
    """
    Extract probable TOC/index headings from OCR text.
    """
    if not text:
        return []
    lines = text.splitlines()
    start_idx = -1
    for i, ln in enumerate(lines[:1200]):
        low = (ln or "").strip().lower()
        if low in {"contents", "table of contents", "index"} or low.startswith("table of contents"):
            start_idx = i
            break
    if start_idx == -1:
        return []

    candidates = []
    for ln in lines[start_idx : start_idx + 1200]:
        raw = (ln or "").strip()
        if len(raw) < 4:
            continue
        # Skip obvious page markers/noise.
        if re.match(r"^page\s+\d+\s+of\s+\d+$", raw, flags=re.IGNORECASE):
            continue
        if re.match(r"^-+\s*page\s+\d+\s*-+$", raw, flags=re.IGNORECASE):
            continue
        # Common TOC line with page number tail.
        m = re.match(
            r"^([A-Za-z0-9][A-Za-z0-9\s,().&/\-]{2,160}?)(?:\s+\.{2,}\s*|\s+)(\d{1,4})$",
            raw,
        )
        if m:
            title = re.sub(r"\s+", " ", m.group(1)).strip(" .-")
            if len(title) >= 4:
                candidates.append(title)
            continue
        # Also keep explicit legal heading markers in TOC blocks.
        if re.match(r"^(Section|Article|Chapter|Part|Schedule)\b", raw, flags=re.IGNORECASE):
            candidates.append(re.sub(r"\s+", " ", raw).strip())

    # Dedup while preserving order.
    out = []
    seen = set()
    for c in candidates:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_entries:
            break
    return out


def _heading_type_and_id(title: str) -> tuple[str, str | None]:
    t = (title or "").strip()
    m = re.match(r"^(Section|Article|Chapter|Part|Schedule|Clause|Para(?:graph)?)\s+([0-9A-Za-z\-\(\)\.]+)", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2)
    # Numbered heading like "151. ..."
    m2 = re.match(r"^([0-9A-Za-z]+)\.\s+", t)
    if m2:
        return "section", m2.group(1)
    return "section", None


def chunk_by_toc(text: str, toc_entries: list[str]) -> list[dict]:
    """
    Build chunks using TOC headings detected in the same text.
    Falls back to [] when reliable heading anchors are not found.
    """
    if not text or not toc_entries:
        return []

    hits = []
    for heading in toc_entries[:120]:
        # Flexible whitespace match.
        parts = [p for p in re.split(r"\s+", heading.strip()) if p]
        if len(parts) < 2:
            continue
        pattern = r"(?im)^\s*" + r"[\s\-–—]*".join(re.escape(p) for p in parts[:10]) + r"\b.*$"
        m = re.search(pattern, text)
        if not m:
            continue
        hits.append((m.start(), m.end(), m.group(0).strip()))

    if len(hits) < 5:
        return []

    # Dedup close/overlapping hits and order by position.
    hits.sort(key=lambda x: x[0])
    dedup = []
    for h in hits:
        if dedup and h[0] - dedup[-1][0] < 20:
            continue
        dedup.append(h)

    chunks = []
    for i, (st, en, title_line) in enumerate(dedup):
        nxt = dedup[i + 1][0] if i + 1 < len(dedup) else len(text)
        block = text[st:nxt].strip()
        if len(block) < 40:
            continue
        section_type, section_id = _heading_type_and_id(title_line)
        chunks.append(
            {
                "section_id": section_id,
                "section_type": section_type,
                "section_title": title_line[:240],
                "text": block[:2000],
                "full_text": block,
                "has_section_marker": True,
                "chunk_strategy": "toc",
            }
        )

    # Require useful minimum to avoid bad splits.
    if len(chunks) < 4:
        return []
    return chunks


def extract_section_from_question(question: str) -> tuple[str | None, str | None]:
    """
    Extract section number and type from a question.
    
    E.g. "What is Section 151?" → ("151", "section")
    E.g. "Define Article 12 of the Constitution" → ("12", "article")
    
    Returns: (section_number, section_type) or (None, None)
    Note: Returns only the FIRST section found. For multiple sections, use extract_all_sections()
    """
    
    # Pattern: "Section 123", "Article 456", etc.
    pattern = r'(?:Section|Article|Chapter|Part|Schedule|Clause|Para(?:graph)?)\s+([0-9A-Za-z]+(?:\([0-9A-Za-z]+\))*)'
    
    match = re.search(pattern, question, re.IGNORECASE)
    
    if not match:
        return None, None
    
    section_num = match.group(1).strip()
    section_type_match = re.match(r'(Section|Article|Chapter|Part|Schedule|Clause|Para(?:graph)?)', match.group(0), re.IGNORECASE)
    section_type = section_type_match.group(1).lower() if section_type_match else "section"
    
    return section_num, section_type


def extract_all_sections(question: str) -> list[tuple[str, str]]:
    """
    Extract ALL section numbers and types from a question.
    
    E.g. "What is Section 151 and 152?" → [("151", "section"), ("152", "section")]
    E.g. "Articles 12 and 13 of Constitution" → [("12", "article"), ("13", "article")]
    E.g. "Sections 151, 152 and 153" → [("151", "section"), ("152", "section"), ("153", "section")]
    
    Returns: List of (section_number, section_type) tuples
    """
    
    sections = []
    seen = set()
    
    # Pattern to match pluralized keywords with comma/and-separated numbers
    # Handles both singular (Section 151) and plural (Sections 151, 152 and 153)
    # Added 's?' to keywords to handle plural forms
    # Changed number pattern to: first_number + (comma/and + more_numbers)*
    plural_pattern = r'(?:Sections?|Articles?|Chapters?|Parts?|Schedules?|Clauses?|Para(?:graph)?s?)\s+([0-9A-Za-z]+(?:\([0-9A-Za-z]+\))*(?:(?:\s*[,;]\s*|\s+and\s+)[0-9A-Za-z]+(?:\([0-9A-Za-z]+\))*)*)'
    
    for match in re.finditer(plural_pattern, question, re.IGNORECASE):
        numbers_text = match.group(1)
        
        # Extract keyword type from the full matched string
        keyword_match = re.match(r'(Section|Article|Chapter|Part|Schedule|Clause|Para(?:graph)?)', match.group(0), re.IGNORECASE)
        section_type = keyword_match.group(1).lower() if keyword_match else "section"
        
        # Extract all individual numbers from the matched numbers_text
        # E.g. "151, 152 and 153" → ["151", "152", "153"]
        for num in re.findall(r'[0-9A-Za-z]+(?:\([0-9A-Za-z]+\))*', numbers_text):
            key = (num, section_type)
            if key not in seen:
                sections.append(key)
                seen.add(key)

    # Fallback for singular forms that include nested markers, e.g. "Section 21(1)(a)".
    singular_pattern = r'(?:Section|Article|Chapter|Part|Schedule|Clause|Para(?:graph)?)\s+([0-9A-Za-z]+(?:\([0-9A-Za-z]+\))*)'
    for match in re.finditer(singular_pattern, question, re.IGNORECASE):
        token = match.group(1)
        keyword_match = re.match(r'(Section|Article|Chapter|Part|Schedule|Clause|Para(?:graph)?)', match.group(0), re.IGNORECASE)
        section_type = keyword_match.group(1).lower() if keyword_match else "section"
        key = (token, section_type)
        if key not in seen:
            sections.append(key)
            seen.add(key)

    return sections
