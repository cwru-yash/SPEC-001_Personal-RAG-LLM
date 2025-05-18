# src/pipeline/extractors/document_extractor.py
from typing import Dict, Any
from src.models.document import Document

class DocumentExtractor:
    """Specialized extractor for document-type PDFs (articles, reports, etc.)."""
    
    def extract(self, pdf, base_document: Document) -> Document:
        """Extract text and metadata optimized for document-type PDFs."""
        text_content = ""
        
        # Extract text from all pages with paragraph preservation
        for page in pdf:
            page_text = page.get_text()
            text_content += page_text + "\n\n"
            
        # Clean up extra whitespace
        text_content = self._clean_text(text_content)
        base_document.text_content = text_content
        
        # Extract document-specific metadata
        base_document.metadata["document_type"] = self._detect_document_type(text_content)
        
        # Look for sections and headings
        sections = self._extract_sections(text_content)
        if sections:
            base_document.metadata["sections"] = sections
            
        return base_document
    
    def _clean_text(self, text: str) -> str:
        """Clean up text by removing extra whitespace, etc."""
        # Replace multiple newlines with double newline
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing/leading whitespace
        text = text.strip()
        
        return text
    
    def _detect_document_type(self, text: str) -> str:
        """Attempt to detect the specific document type."""
        import re
        text_lower = text.lower()
        
        # Check for common document types
        if re.search(r'\breport\b', text_lower):
            return "report"
        elif re.search(r'\barticle\b', text_lower) or re.search(r'\babstract\b', text_lower):
            return "article"
        elif re.search(r'\bpaper\b', text_lower):
            return "paper"
        elif re.search(r'\bmanual\b', text_lower) or re.search(r'\bguide\b', text_lower):
            return "manual"
        elif re.search(r'\bcontract\b', text_lower) or re.search(r'\bagreement\b', text_lower):
            return "contract"
        else:
            return "generic"
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections and headings from document."""
        import re
        
        # Look for section patterns like "1. Introduction" or "Chapter 1:"
        section_patterns = [
            r'^\s*(\d+\.?\s+[A-Z][a-zA-Z ]+)$',  # Numbered sections (1. Introduction)
            r'^\s*([A-Z][A-Z ]+):?\s*$',         # ALL CAPS headings
            r'^\s*(Chapter \d+:?\s+.+)$',        # Chapter headings
            r'^\s*([A-Z][a-zA-Z ]+)\s*$'         # Title Case headings at line start
        ]
        
        sections = {}
        current_section = "Header"
        section_text = []
        
        for line in text.split('\n'):
            is_heading = False
            
            # Check if line is a heading
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous section
                    if section_text:
                        sections[current_section] = '\n'.join(section_text)
                    
                    # Start new section
                    current_section = match.group(1).strip()
                    section_text = []
                    is_heading = True
                    break
            
            if not is_heading:
                section_text.append(line)
        
        # Save final section
        if section_text:
            sections[current_section] = '\n'.join(section_text)
            
        return sections