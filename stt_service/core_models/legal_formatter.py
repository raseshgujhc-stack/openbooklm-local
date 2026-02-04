import re
from datetime import datetime
from typing import List, Dict, Optional

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
        
        # Common Indian judiciary patterns
        self.case_number_pattern = r'([A-Z]{1,4}\s?\d+\s?/\s?\d{4})'
        self.citation_pattern = r'(\d{4})\s*(?:\(?\d*\)?)?\s*(?:AIR|SCC|SCR|JT|SCALE)\s+([A-Z]{1,4})\s+(\d+)'
    
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

        paragraphs = text.split(". ")
        for i, para in enumerate(paragraphs):
            if para.strip():
                doc += f"{i+1}. {para.strip()}"
                if not para.endswith("."):
                    doc += "."
                doc += "\n\n"

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
        
        # Main content with paragraphs
        paragraphs = text.split('. ')
        for i, para in enumerate(paragraphs):
            if para.strip():
                # Add paragraph number for legal documents
                para_text = f"{i+1}. {para.strip()}"
                if not para.endswith('.'):
                    para_text += '.'
                doc += f"{para_text}\n\n"
        
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
        # Similar structure with Supreme Court formatting
        pass
    
    def format_district_court(self, text: str, metadata: Dict, filename: str, chunks: List[Dict]) -> str:
        """Format as District Court order"""
        pass
    
    def _clean_transcript(self, text: str) -> str:
        """Clean transcription text"""
        # Remove filler words
        filler_words = [
            'um', 'uh', 'like', 'you know', 'actually', 'basically',
            'kind of', 'sort of', 'I mean', 'well', 'so', 'okay'
        ]
        
        for word in filler_words:
            text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Capitalize sentences
        sentences = re.split(r'([.!?])\s+', text)
        cleaned = ''
        for i in range(0, len(sentences)-1, 2):
            sentence = sentences[i].strip()
            punctuation = sentences[i+1] if i+1 < len(sentences) else ''
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if sentence else ''
                cleaned += sentence + punctuation + ' '
        
        return cleaned.strip()
    
    def _correct_legal_terms(self, text: str) -> str:
        """Correct common legal term mis-transcriptions"""
        for category in self.legal_terms.values():
            for wrong, correct in category.items():
                # Case insensitive replacement
                text = re.sub(rf'\b{re.escape(wrong)}\b', correct, text, flags=re.IGNORECASE)
        
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
