"""
Legal transcript formatter for Indian judiciary output styles.

Includes lightweight real-time formatting and richer final document formatting
for High Court, Supreme Court, District Court, and tribunal patterns.
"""

import re
import csv
import os
from datetime import datetime
from typing import List, Dict, Optional
from utils.correction_feedback import load_feedback_phrase_map

class LegalFormatter:
    """Format transcripts for Indian judiciary"""
    
    def __init__(self):
        self.court_formats = {
            "supreme_court": self.format_supreme_court,
            "high_court": self.format_high_court,
            "district_court": self.format_district_court,
            "tribunal": self.format_tribunal
        }
        
        # Indian legal terminology database
        self.legal_terms = self._load_legal_terms()
        self.act_alias_map, self.act_year_map = self._load_legislation_catalog()
        self.act_alias_patterns = self._compile_act_alias_patterns(self.act_alias_map)
        self.correction_rules = self._load_correction_rules()
        self.feedback_phrase_map = load_feedback_phrase_map()
        
        # Common Indian judiciary patterns
        self.case_number_pattern = r'([A-Z]{1,4}\s?\d+\s?/\s?\d{4})'
        self.citation_pattern = r'(\d{4})\s*(?:\(?\d*\)?)?\s*(?:AIR|SCC|SCR|JT|SCALE)\s+([A-Z]{1,4})\s+(\d+)'

    def format_realtime(self, text: str, court_type: Optional[str] = None) -> str:
        """Lightweight formatter for streaming chunks.

        Keeps latency low while still applying legal-term normalization.
        """
        if not text:
            return ""

        cleaned = self._clean_transcript(text)
        corrected = self._correct_legal_terms(cleaned)
        corrected = self._format_citations(corrected)

        if court_type == "supreme_court":
            return f"[SC] {corrected}"
        if court_type == "district_court":
            return f"[DC] {corrected}"
        if court_type == "tribunal":
            return f"[TRIBUNAL] {corrected}"
        return corrected
    
    def _load_legal_terms(self) -> Dict:
        """Load Indian legal terminology"""
        return {
            "common": {
                "res judicata": "res judicata",
                "locus standi": "locus standi",
                "stare decisis": "stare decisis",
                "amicus curiae": "amicus curiae",
                "habeas corpus": "habeas corpus",
                "mandamus": "mandamus",
                "certiorari": "certiorari",
                "quo warranto": "quo warranto",
                "obiter dicta": "obiter dicta",
                "ratio decidendi": "ratio decidendi",
                "sub judice": "sub judice",
                "ex parte": "ex parte",
                "in personam": "in personam",
                "in rem": "in rem",
                "prima facie": "prima facie",
                "ad interim": "ad interim",
            },
            "indian_specific": {
                "order 39": "Order XXXIX",
                "order 41": "Order XLI",
                "order 7 rule 11": "Order VII Rule 11",
                "section 482": "Section 482",
                "cpc": "CPC",
                "cpc1908": "CPC, 1908",
                "crpc": "CrPC",
                "crpc1973": "CrPC, 1973",
                "ipc": "IPC",
                "ipc1860": "IPC, 1860",
                "constitution": "Constitution of India",
                "article 226": "Article 226",
                "article 32": "Article 32",
                "article 21": "Article 21",
            }
        }

    def _normalize_alias_key(self, value: str) -> str:
        norm = re.sub(r"[^A-Za-z0-9\s]+", " ", (value or "").strip().lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        return norm

    def _canonicalize_act_name(self, name: str) -> str:
        clean = " ".join((name or "").split()).strip()
        if not clean:
            return ""
        return clean.title()

    def _load_legislation_catalog(self):
        """
        Load act aliases and year metadata from CSV catalog.
        Data-driven source (no hardcoded act list).
        """
        alias_map: Dict[str, str] = {}
        year_map: Dict[str, str] = {}

        candidate_paths = []
        env_path = os.getenv("LEGISLATION_CSV_PATH")
        if env_path:
            candidate_paths.append(env_path)
        candidate_paths.extend(
            [
                "/app/legislation/legal_act_sections_refined.csv",
                "/app/legislation/structured_legislation_v4.csv",
            ]
        )

        selected_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                selected_path = p
                break

        if not selected_path:
            return alias_map, year_map

        try:
            with open(selected_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    act_name = (row.get("clean_act_name") or row.get("act_name") or "").strip()
                    if not act_name:
                        continue

                    canonical = self._canonicalize_act_name(act_name)
                    if not canonical:
                        continue

                    year = str(row.get("act_year") or "").strip()
                    if re.fullmatch(r"(18|19|20)\d{2}", year):
                        year_map.setdefault(canonical, year)

                    raw_variants = {
                        canonical,
                        act_name,
                        (row.get("act_name") or "").strip(),
                    }
                    for variant in list(raw_variants):
                        if variant.lower().startswith("the "):
                            raw_variants.add(variant[4:])

                    for alias in raw_variants:
                        key = self._normalize_alias_key(alias)
                        if not key:
                            continue
                        alias_map.setdefault(key, canonical)
        except Exception:
            return {}, {}

        return alias_map, year_map

    def _load_correction_rules(self):
        """
        Load ASR correction rules from CSV "table" to avoid code hardcoding.

        CSV columns:
        - pattern: regex pattern
        - replacement: replacement text
        - scope: legal|global (default legal)
        - requires_any: optional pipe-separated terms that must exist in text
        - priority: integer (higher runs first)
        - enabled: true|false
        """
        default_path = "/app/legislation/legal_asr_corrections.csv"
        candidate_paths = []
        env_path = os.getenv("LEGAL_ASR_RULES_CSV")
        if env_path:
            candidate_paths.append(env_path)
        candidate_paths.extend(
            [
                default_path,
                "/home/ubuntu/openbooklm-local/legal_asr_corrections.csv",
                os.path.join(os.path.dirname(__file__), "..", "..", "legal_asr_corrections.csv"),
            ]
        )
        path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
        if not path:
            return []

        rules = []
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    enabled_raw = str(row.get("enabled", "true")).strip().lower()
                    if enabled_raw in {"0", "false", "no", "off"}:
                        continue

                    pattern = (row.get("pattern") or "").strip()
                    replacement = (row.get("replacement") or "").strip()
                    if not pattern:
                        continue

                    scope = (row.get("scope") or "legal").strip().lower()
                    priority_raw = str(row.get("priority") or "100").strip()
                    try:
                        priority = int(priority_raw)
                    except Exception:
                        priority = 100

                    requires_any_raw = (row.get("requires_any") or "").strip()
                    requires_any = [
                        token.strip().lower()
                        for token in requires_any_raw.split("|")
                        if token.strip()
                    ]

                    try:
                        compiled = re.compile(pattern, flags=re.IGNORECASE)
                    except re.error:
                        continue

                    rules.append(
                        {
                            "pattern": compiled,
                            "replacement": replacement,
                            "scope": scope,
                            "requires_any": requires_any,
                            "priority": priority,
                        }
                    )
        except Exception:
            return []

        rules.sort(key=lambda r: r["priority"], reverse=True)
        return rules

    def _legal_context_confidence(self, text: str) -> float:
        """
        Heuristic confidence that the text is legal-context dictation.
        Used to avoid over-corrections on non-legal speech.
        """
        lower = (text or "").lower()
        anchors = [
            "section", "article", "act", "judge", "magistrate", "appeal",
            "accused", "complainant", "criminal case", "bnss", "crpc", "ipc", "cpc",
        ]
        hits = sum(1 for token in anchors if token in lower)
        return min(1.0, hits / 4.0)

    def _compile_act_alias_patterns(self, alias_map: Dict[str, str]):
        patterns = []
        for alias_key, canonical in alias_map.items():
            # Keep only legal-statute-like aliases for safer replacement.
            if not any(token in alias_key for token in [" act", " code", " regulation", " rules"]):
                continue
            if len(alias_key) < 6:
                continue

            alias_regex = re.escape(alias_key).replace(r"\ ", r"\s+")
            pattern = re.compile(rf"\b{alias_regex}\b", flags=re.IGNORECASE)
            patterns.append((pattern, canonical))

        # Longest aliases first to avoid partial replacement.
        patterns.sort(key=lambda item: len(item[0].pattern), reverse=True)
        return patterns
    
    def format_chunk(self, text: str, chunk_number: int, time_start: float, time_end: float) -> str:
        """Format a single chunk of transcription"""
        # Clean text
        text = self._clean_transcript(text)
        
        # Apply legal term corrections
        text = self._correct_legal_terms(text)
        
        # Format as chunk
        formatted = f"\n{'='*80}\n"
        formatted += f"CHUNK {chunk_number:03d} | Time: {time_start:.1f}s - {time_end:.1f}s\n"
        formatted += f"{'='*80}\n\n"
        formatted += self._format_paragraphs(text)
        formatted += f"\n\n"
        
        return formatted
    
    def format_complete_document(self, full_text: str, chunks: List[Dict], filename: str) -> Dict:
        """Format complete transcription as Indian judiciary document"""
        
        # Extract metadata
        metadata = self._extract_metadata(full_text)
        
        # Choose court format based on content
        court_type = self._detect_court_type(full_text)
        
        # Format document
        document = self.court_formats[court_type](
            full_text, 
            metadata, 
            filename,
            chunks
        )
        
        return {
            "document": document,
            "metadata": metadata,
            "court_type": court_type,
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "total_chunks": len(chunks)
        }
        
    def format_tribunal(self, text: str, metadata: dict, filename: str, chunks: list) -> str:
        """Format as Tribunal order (fallback/simple format)"""

        text = self._clean_transcript(text)
        text = self._correct_legal_terms(text)

        doc = f"{'='*100}\n"
        doc += "BEFORE THE HONOURABLE TRIBUNAL\n"
        doc += f"{'='*100}\n\n"

        if metadata.get("case_number"):
            doc += f"Case No.: {metadata['case_number']}\n\n"

        doc += "ORDER\n\n"

        doc += self._format_paragraphs(text)

        doc += f"{'='*100}\n"
        doc += "DICTATED AND TRANSCRIBED BY JUDICIAL STT SYSTEM\n"
        doc += f"{'='*100}\n"

        return doc
    
    def format_high_court(self, text: str, metadata: Dict, filename: str, chunks: List[Dict]) -> str:
        """Format as High Court judgment/order"""
        
        # Clean and structure text
        text = self._clean_transcript(text)
        text = self._correct_legal_terms(text)
        text = self._format_citations(text)
        
        # Build document
        doc = f"{'='*100}\n"
        doc += f"IN THE HIGH COURT OF GUJARAT AT AHMEDABAD\n"
        doc += f"{'='*100}\n\n"
        
        # Case number (if detected)
        if metadata.get('case_number'):
            doc += f"Case No.: {metadata['case_number']}\n"
        
        doc += f"Date of Dictation: {datetime.now().strftime('%d %B, %Y')}\n\n"
        doc += f"{'─'*100}\n\n"
        
        # Add transcription source
        doc += f"TRANSCRIPTION OF DICTATION\n"
        doc += f"Source Audio: {filename}\n"
        doc += f"Transcribed On: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n"
        doc += f"Total Duration: {sum(c.get('duration', 0) for c in chunks):.1f} seconds\n"
        doc += f"Number of Chunks: {len(chunks)}\n\n"
        doc += f"{'─'*100}\n\n"
        
        # Main content in narrative legal paragraphs (do not force sentence numbering).
        doc += self._format_paragraphs(text)
        
        # Add conclusion
        doc += f"\n{'─'*100}\n\n"
        doc += "DICTATED, TRANSCRIBED AND PRINTED BY:\n"
        doc += "JUDICIAL STT SYSTEM\n"
        doc += "OpenBookLM Judicial Transcription Service\n\n"
        
        doc += f"{'='*100}\n"
        doc += "END OF TRANSCRIPTION\n"
        doc += f"{'='*100}\n"
        
        return doc
    
    def format_supreme_court(self, text: str, metadata: Dict, filename: str, chunks: List[Dict]) -> str:
        """Format as Supreme Court judgment"""
        text = self._clean_transcript(text)
        text = self._correct_legal_terms(text)
        text = self._format_citations(text)

        doc = f"{'='*100}\n"
        doc += "IN THE SUPREME COURT OF INDIA\n"
        doc += f"{'='*100}\n\n"
        if metadata.get('case_number'):
            doc += f"Case No.: {metadata['case_number']}\n"
        doc += f"Date of Dictation: {datetime.now().strftime('%d %B, %Y')}\n\n"
        doc += f"{'─'*100}\n\n"
        doc += self._format_paragraphs(text)
        doc += f"{'='*100}\nEND OF TRANSCRIPTION\n{'='*100}\n"
        return doc

    def format_district_court(self, text: str, metadata: Dict, filename: str, chunks: List[Dict]) -> str:
        """Format as District Court order"""
        text = self._clean_transcript(text)
        text = self._correct_legal_terms(text)

        doc = f"{'='*100}\n"
        doc += "IN THE DISTRICT COURT\n"
        doc += f"{'='*100}\n\n"
        if metadata.get('case_number'):
            doc += f"Case No.: {metadata['case_number']}\n\n"
        doc += "ORDER\n\n"
        doc += self._format_paragraphs(text)
        doc += f"{'='*100}\nEND OF TRANSCRIPTION\n{'='*100}\n"
        return doc
    
    def _clean_transcript(self, text: str) -> str:
        """Clean transcription text"""
        if not text:
            return ""

        # Remove filler words
        filler_words = [
            'um', 'uh', 'like', 'you know', 'actually', 'basically',
            'kind of', 'sort of', 'I mean', 'well', 'so', 'okay'
        ]
        
        for word in filler_words:
            text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)

        # Convert spoken punctuation/structure commands into symbols.
        text = self._apply_dictation_commands(text)
        
        # Normalize whitespace and punctuation artifacts from ASR.
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([.?!])\1+", r"\1", text)
        text = re.sub(r"\s+([.?!,;:])", r"\1", text)
        text = re.sub(r"([.?!,;:])([A-Za-z])", r"\1 \2", text)
        text = re.sub(r"¶\s*[.]+", "¶", text)
        
        # Split into sentence-like fragments; keep unpunctuated chunks too.
        fragments = [frag.strip() for frag in re.split(r'(?<=[.!?])\s+', text) if frag.strip()]
        if not fragments:
            return text.strip()

        normalized: List[str] = []
        for frag in fragments:
            if frag:
                normalized.append(frag[0].upper() + frag[1:] if len(frag) > 1 else frag.upper())
        return " ".join(normalized).strip()

    def _apply_dictation_commands(self, text: str) -> str:
        """Apply spoken punctuation and structural cues from dictation."""
        out = text

        # Normalize literal marker variants users may type/paste manually.
        out = re.sub(r"\[\[\s*table\s*_?\s*start\s*\]\]", " [[TABLE_START]] ", out, flags=re.IGNORECASE)
        out = re.sub(r"\[\[\s*table\s*_?\s*col(?:umn)?\s*\]\]", " [[TABLE_COL]] ", out, flags=re.IGNORECASE)
        out = re.sub(r"\[\[\s*table\s*_?\s*row\s*\]\]", " [[TABLE_ROW]] ", out, flags=re.IGNORECASE)
        out = re.sub(r"\[\[\s*table\s*_?\s*end\s*\]\]", " [[TABLE_END]] ", out, flags=re.IGNORECASE)

        # Bracket commands (keep generic).
        out = re.sub(r"\bopen\s+bracket\b", " (", out, flags=re.IGNORECASE)
        out = re.sub(r"\bin bracket\b", " (", out, flags=re.IGNORECASE)
        out = re.sub(r"\b(?:close|closed|closing)\s+bracket\b", ") ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bbracket\s+(?:close|closed|rose)\b", ") ", out, flags=re.IGNORECASE)

        # Spoken punctuation tokens.
        token_map = {
            r"\bcomma\b": ",",
            r"\bfull\s+stop\b": ".",
            r"\bcolon\b": ":",
            r"\bsemi\s*colon\b": ";",
            r"\bquestion\s+mark\b": "?",
        }
        for pattern, symbol in token_map.items():
            out = re.sub(pattern, symbol, out, flags=re.IGNORECASE)

        # Spoken slash patterns used in case numbers.
        out = re.sub(r"\bslas\b", "slash", out, flags=re.IGNORECASE)
        out = re.sub(r"\bslash\b", "/", out, flags=re.IGNORECASE)
        out = re.sub(r"\bpoint\s+number\s+one\b", "1.", out, flags=re.IGNORECASE)
        out = re.sub(r"\bpoint\s+number\s+two\b", "2.", out, flags=re.IGNORECASE)
        out = re.sub(r"\bpoint\s+number\s+three\b", "3.", out, flags=re.IGNORECASE)
        out = re.sub(r"\bnext\s+point\b", " ¶ ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bnew\s+paragraph\b", " ¶ ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bnext\s+paragraph\b", " ¶ ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bnext\s+para\b", " ¶ ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bnext\s+pera\b", " ¶ ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bnew\s+line\b", "\n", out, flags=re.IGNORECASE)
        out = re.sub(r"\bnext\s+line\b", "\n", out, flags=re.IGNORECASE)

        # Dictated table directives.
        out = re.sub(r"\btable\s+first\s+column\b", " [[TABLE_START]] ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bfirst\s+column\b", " [[TABLE_COL]] ", out, flags=re.IGNORECASE)
        out = re.sub(r"\b(?:start|begin|in)\s+table\b", " [[TABLE_START]] ", out, flags=re.IGNORECASE)
        out = re.sub(r"\btable\s+start\b", " [[TABLE_START]] ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bstart\s+table\b", " [[TABLE_START]] ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bnext\s+column\b", " [[TABLE_COL]] ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bnext\s+cell\b", " [[TABLE_COL]] ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bcell\s+next\b", " [[TABLE_COL]] ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bnext\s+row\b", " [[TABLE_ROW]] ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bend\s+table\b", " [[TABLE_END]] ", out, flags=re.IGNORECASE)
        out = self._ensure_table_start_if_missing(out)
        out = self._ensure_table_end_if_missing(out)
        out = self._render_dictated_tables(out)

        # Clean spacing around punctuation symbols introduced above.
        out = re.sub(r"\s+([,.;:?/)\]])", r"\1", out)
        out = re.sub(r"([(/])\s+", r"\1", out)
        out = re.sub(r"\(\s+", "(", out)
        out = re.sub(r"\s+\)", ")", out)
        return out

    def _render_dictated_tables(self, text: str) -> str:
        """
        Convert spoken table commands into markdown-table output.
        """
        block_re = re.compile(r"\[\[TABLE_START\]\](.*?)\[\[TABLE_END\]\]", flags=re.DOTALL)

        def _block_to_table(block: str) -> str:
            rows = []
            for row_blob in block.split("[[TABLE_ROW]]"):
                cells = [c.strip(" .,\n\t") for c in row_blob.split("[[TABLE_COL]]")]
                cells = [c for c in cells if c]
                if cells:
                    rows.append(cells)
            if not rows:
                return ""

            width = max(len(r) for r in rows)
            norm_rows = [r + [""] * (width - len(r)) for r in rows]
            header = norm_rows[0]
            lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * width) + " |",
            ]
            for row in norm_rows[1:]:
                lines.append("| " + " | ".join(row) + " |")
            return "\n" + "\n".join(lines) + "\n"

        return block_re.sub(lambda m: _block_to_table(m.group(1)), text)

    def _ensure_table_start_if_missing(self, text: str) -> str:
        """
        If row/col/end tokens exist but start token is missing, infer table start.
        """
        out = text
        has_col_or_row = ("[[TABLE_COL]]" in out) or ("[[TABLE_ROW]]" in out)
        has_end = "[[TABLE_END]]" in out
        has_start = "[[TABLE_START]]" in out
        if has_col_or_row and has_end and not has_start:
            out = out.replace("[[TABLE_COL]]", "[[TABLE_START]] [[TABLE_COL]]", 1)
        return out

    def _ensure_table_end_if_missing(self, text: str) -> str:
        """
        If a table starts but no explicit end marker is provided, close it at end.
        """
        out = text
        has_start = "[[TABLE_START]]" in out
        has_end = "[[TABLE_END]]" in out
        has_col_or_row = ("[[TABLE_COL]]" in out) or ("[[TABLE_ROW]]" in out)
        if has_start and has_col_or_row and not has_end:
            out = out.rstrip() + " [[TABLE_END]]"
        return out
    
    def _correct_legal_terms(self, text: str) -> str:
        """Correct common legal term mis-transcriptions"""
        text = self._apply_feedback_phrase_corrections(text)
        for category in self.legal_terms.values():
            for wrong, correct in category.items():
                # Case insensitive replacement
                text = re.sub(rf'\b{re.escape(wrong)}\b', correct, text, flags=re.IGNORECASE)

        text = self._normalize_legal_phrases(text)
        text = self._normalize_numeric_legal_patterns(text)
        text = self._normalize_designations(text)
        text = self._normalize_statutory_references(text)
        return text

    def reload_feedback_phrase_map(self):
        """Refresh dynamically learned phrase corrections from user feedback."""
        self.feedback_phrase_map = load_feedback_phrase_map()

    def normalize_user_edit(self, text: str) -> str:
        """
        Apply dictation commands for user-edited transcript text without
        heavy legal reformatting.
        """
        out = (text or "").strip()
        # Ignore common edit-intent prefixes spoken or typed by users.
        out = re.sub(r"^(?:edit|correct|update)\s*[:\-]?\s+", "", out, flags=re.IGNORECASE)
        out = self._apply_dictation_commands(out)
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        out = re.sub(r" *\n *", "\n", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out.strip()

    def _apply_feedback_phrase_corrections(self, text: str) -> str:
        out = text
        if not self.feedback_phrase_map:
            return out
        pairs = sorted(self.feedback_phrase_map.items(), key=lambda kv: len(kv[0]), reverse=True)
        for wrong, correct in pairs[:300]:
            if len(wrong) < 3:
                continue
            out = re.sub(rf"\b{re.escape(wrong)}\b", correct, out, flags=re.IGNORECASE)
        return out

    def _normalize_numeric_legal_patterns(self, text: str) -> str:
        """Repair numeric drifts in common legal constructs without paragraph-specific rules."""
        out = text
        # Example drift: "delay of 1 and 92 days" -> "delay of 192 days"
        out = re.sub(
            r"\b(delay\s+of)\s+([0-9])\s+(?:and|n)\s+([0-9]{1,4})\s+days\b",
            lambda m: f"{m.group(1)} {m.group(2)}{m.group(3)} days",
            out,
            flags=re.IGNORECASE,
        )
        # Collapse spaced digit runs in case references: "3 9 2 / 2 0 2 3"
        out = re.sub(
            r"\b((?:\d\s+){2,}\d)\b",
            lambda m: re.sub(r"\s+", "", m.group(1)),
            out,
        )
        return out

    def _normalize_legal_phrases(self, text: str) -> str:
        """
        Apply correction rules loaded from external CSV "table".
        Rules can be scope-aware and context-gated.
        """
        out = text
        context_conf = self._legal_context_confidence(out)
        lower_text = out.lower()

        for rule in self.correction_rules:
            scope = rule.get("scope", "legal")
            if scope == "legal" and context_conf < 0.25:
                continue

            required = rule.get("requires_any") or []
            if required and not any(token in lower_text for token in required):
                continue

            out = rule["pattern"].sub(rule["replacement"], out)

        # Keep generic structural cleanup independent of external rules.
        out = re.sub(
            r"\bCase\s+No\.\s*(\d{1,6})\s*/\s*(\d{2,4})\b",
            lambda m: f"Case No. {m.group(1)}/{m.group(2)}",
            out,
            flags=re.IGNORECASE,
        )
        return out

    def _normalize_designations(self, text: str) -> str:
        """Capitalize known judicial designations consistently."""
        designations = [
            "Additional Public Prosecutor",
            "Public Prosecutor",
            "Additional Chief Judicial Magistrate",
            "Chief Judicial Magistrate",
            "Additional Senior Civil Judge",
            "Senior Civil Judge",
            "Civil Judge",
            "Judicial Magistrate",
            "Sessions Judge",
            "District Judge",
        ]
        out = text
        for title in designations:
            out = re.sub(
                rf"\b{re.escape(title.lower())}\b",
                title,
                out,
                flags=re.IGNORECASE,
            )

        # Convert ordinal words for common designation prefix.
        ordinal_map = {
            "first": "1st",
            "second": "2nd",
            "third": "3rd",
            "fourth": "4th",
            "fifth": "5th",
            "sixth": "6th",
            "seventh": "7th",
            "eighth": "8th",
            "ninth": "9th",
            "tenth": "10th",
        }
        for word, numeral in ordinal_map.items():
            out = re.sub(
                rf"\blearned\s+{word}\s+additional\b",
                f"learned {numeral} Additional",
                out,
                flags=re.IGNORECASE,
            )

        # Normalize leading descriptor `learned` with designation.
        out = re.sub(r"\blearned\s+([A-Z][A-Za-z ]+Judge|[A-Z][A-Za-z ]+Magistrate)\b", r"learned \1", out)
        return out

    def _normalize_statutory_references(self, text: str) -> str:
        """Normalize statutory references: Section casing and Act, YEAR formatting."""

        # Section / Article capitalization with preserved number token.
        text = re.sub(
            r"\bsection\s+([0-9]+[A-Za-z\-]*)\b",
            lambda m: f"Section {m.group(1)}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(Section\s+[0-9]+[A-Za-z\-]*)\s+Of\b",
            r"\1 of",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\barticle\s+([0-9]+[A-Za-z\-]*)\b",
            lambda m: f"Article {m.group(1)}",
            text,
            flags=re.IGNORECASE,
        )

        # Replace act aliases from data catalog with canonical names.
        if self.act_alias_patterns and "act" in text.lower():
            normalized = text
            for pattern, canonical in self.act_alias_patterns:
                normalized = pattern.sub(canonical, normalized)
            text = normalized

        # Ensure '<Act Name> Act, YYYY' style formatting.
        def _act_year_repl(match: re.Match) -> str:
            act_name = " ".join(match.group(1).split()).title()
            year = match.group(2)
            return f"{act_name}, {year}"

        text = re.sub(
            r"\b([A-Za-z][A-Za-z\s]{2,}? Act)\s*,?\s*((?:18|19|20)\d{2})\b",
            _act_year_repl,
            text,
            flags=re.IGNORECASE,
        )

        # If catalog has canonical year, enforce comma-year format.
        for canonical, year in self.act_year_map.items():
            text = re.sub(
                rf"\b{re.escape(canonical)}(?:\s*,?\s*(?:{re.escape(year)}))?\b",
                f"{canonical}, {year}",
                text,
                flags=re.IGNORECASE,
            )

        return text
    
    def _format_citations(self, text: str) -> str:
        """Format legal citations properly"""
        # Format Indian legal citations
        patterns = [
            # AIR citations
            (r'(\d{4})\s*AIR\s*([A-Z]{2})\s*(\d+)', r'\1 AIR \2 \3'),
            # SCC citations
            (r'(\d{4})\s*(\d+)\s*SCC\s*(\d+)', r'\1 (\2) SCC \3'),
            # Supreme Court cases
            (r'(\d{4})\s*(\d+)\s*SCR\s*(\d+)', r'\1 (\2) SCR \3'),
            # JT citations
            (r'(\d{4})\s*(\d+)\s*JT\s*(\d+)', r'\1 (\2) JT \3'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _extract_metadata(self, text: str) -> Dict:
        """Extract metadata from transcription"""
        metadata = {
            'case_number': None,
            'citations': [],
            'judge_mentioned': False,
            'legal_terms_found': []
        }
        
        # Find case numbers
        case_match = re.search(self.case_number_pattern, text)
        if case_match:
            metadata['case_number'] = case_match.group(1)
        
        # Find citations
        citations = re.findall(self.citation_pattern, text)
        metadata['citations'] = citations
        
        # Check for judge mentions
        judge_keywords = ['honourable', 'justice', 'judge', 'j.', 'hon ble']
        for keyword in judge_keywords:
            if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
                metadata['judge_mentioned'] = True
                break
        
        # Find legal terms
        for category in self.legal_terms.values():
            for term in category.keys():
                if re.search(rf'\b{term}\b', text, re.IGNORECASE):
                    metadata['legal_terms_found'].append(term)
        
        return metadata
    
    def _detect_court_type(self, text: str) -> str:
        """Detect type of court from content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['supreme court', 'sc', 'apex court']):
            return "supreme_court"
        elif any(word in text_lower for word in ['high court', 'hc', 'gujarat high court']):
            return "high_court"
        elif any(word in text_lower for word in ['district court', 'session court', 'magistrate']):
            return "district_court"
        elif any(word in text_lower for word in ['tribunal', 'nclt', 'nclat', 'itat']):
            return "tribunal"
        else:
            return "high_court"  # Default
    
    def _format_paragraphs(self, text: str) -> str:
        """Format text into proper paragraphs"""
        # Respect explicit dictation cue.
        text = re.sub(r"\bnew paragraph\b", " ¶ ", text, flags=re.IGNORECASE)

        if "¶" in text:
            blocks = [b.strip() for b in text.split("¶") if b.strip()]
            blocks = [re.sub(r"^[.]+", "", b).strip() for b in blocks if b.strip()]
            return "\n\n".join(blocks) + ("\n\n" if blocks else "")

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        formatted = ""
        current_paragraph = []
        char_count = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            current_paragraph.append(sentence)
            char_count += len(sentence)
            
            # Start new paragraph after ~500 chars or at logical breaks
            if char_count > 500 or sentence.strip().endswith(('.', ';')):
                if current_paragraph:
                    formatted += ' '.join(current_paragraph) + '\n\n'
                    current_paragraph = []
                    char_count = 0
        
        # Add remaining sentences
        if current_paragraph:
            formatted += ' '.join(current_paragraph) + '\n\n'
        
        return formatted
