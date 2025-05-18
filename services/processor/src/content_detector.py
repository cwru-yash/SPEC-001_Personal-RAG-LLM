# src/pipeline/content_detector.py
import re
from typing import List

class ContentTypeDetector:
    """Detects the content type within a PDF."""
    
    def detect_content_types(self, sample_pages) -> List[str]:
        """Analyze sample pages to detect content types."""
        content_types = ["pdf"]  # Default type
        
        # Check for email patterns
        if self._is_email(sample_pages):
            content_types.append("email")
            
        # Check for presentation patterns
        if self._is_presentation(sample_pages):
            content_types.append("presentation")
            
        # Check for document patterns (Word document, report, etc.)
        if self._is_document(sample_pages):
            content_types.append("document")
            
        return content_types
    
    def _is_email(self, sample_pages) -> bool:
        """Check if the PDF contains email content."""
        email_patterns = [
            r'From:.*@',
            r'To:.*@',
            r'Sent:.*\d{1,2}[:/]\d{1,2}[:/]\d{2,4}',
            r'Subject:',
            r'CC:',
            r'BCC:',
            r'Reply-To:'
        ]
        
        for page in sample_pages:
            text = page.get_text()
            
            # Check for email headers
            header_matches = 0
            for pattern in email_patterns:
                if re.search(pattern, text):
                    header_matches += 1
            
            # If we find multiple email header patterns, it's likely an email
            if header_matches >= 2:
                return True
                
        return False
    
    def _is_presentation(self, sample_pages) -> bool:
        """Check if the PDF is likely a presentation."""
        # Presentations typically have:
        # 1. Large font sizes for titles
        # 2. Limited text per page
        # 3. Bullet points
        # 4. Many images/graphics
        
        presentation_indicators = 0
        
        for page in sample_pages:
            text = page.get_text()
            
            # Check text length - presentations have less text per page
            if len(text.split()) < 100:
                presentation_indicators += 1
                
            # Check for bullet points
            if text.count('•') > 3 or text.count('-') > 3:
                presentation_indicators += 1
                
            # Check text blocks - presentations have fewer text blocks
            if len(page.get_text("blocks")) < 5:
                presentation_indicators += 1
                
            # Check for large font sizes
            spans = page.get_text("dict")["blocks"]
            has_large_font = False
            for block in spans:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            for span in line["spans"]:
                                if span.get("size", 0) > 14:  # Font size > 14pt
                                    has_large_font = True
                                    break
            
            if has_large_font:
                presentation_indicators += 1
                
        # If majority of indicators are true, it's likely a presentation
        return presentation_indicators >= 2
    
    def _is_document(self, sample_pages) -> bool:
        """Check if the PDF is a regular document (default)."""
        # Since most PDFs are documents, we default to True
        # unless other factors indicate it's not a document
        
        # Documents typically have:
        # 1. Multiple paragraphs
        # 2. Consistent font sizes
        # 3. Relatively dense text
        
        for page in sample_pages:
            text = page.get_text()
            
            # Check for paragraphs - documents have multiple paragraphs
            paragraphs = text.split('\n\n')
            if len(paragraphs) < 2:
                continue
                
            # Check for consistent spacing - documents have consistent line spacing
            lines = text.split('\n')
            if len(lines) < 5:
                continue
                
            # If we get here, at least one page looks like a document
            return True
                
        return False