# src/pipeline/extractors/presentation_extractor.py
from typing import Dict, List, Any
from src.models.document import Document

class PresentationExtractor:
    """Specialized extractor for presentation-type PDFs."""
    
    def extract(self, pdf, base_document: Document) -> Document:
        """Extract text and metadata optimized for presentation-type PDFs."""
        slides = []
        text_content = ""
        
        # Process each page as a slide
        for page_num, page in enumerate(pdf):
            slide_text = page.get_text()
            
            # Add slide marker and number
            slide_header = f"\n\n--- SLIDE {page_num + 1} ---\n\n"
            text_content += slide_header + slide_text
            
            # Extract title and content
            title, content = self._extract_slide_structure(slide_text)
            
            slides.append({
                "slide_num": page_num + 1,
                "title": title,
                "content": content
            })
        
        # Update document
        base_document.text_content = text_content
        base_document.metadata["slides"] = slides
        base_document.metadata["slide_count"] = len(slides)
        
        # Extract presentation title (usually from first slide)
        if slides and slides[0]["title"]:
            base_document.metadata["presentation_title"] = slides[0]["title"]
            
        return base_document
    
    def _extract_slide_structure(self, slide_text: str) -> tuple:
        """Extract slide title and content from text."""
        import re
        
        # Split into lines
        lines = slide_text.strip().split('\n')
        
        # Presentation slides typically have the title as the first 1-3 lines
        # with larger font - we'll take a simple approach
        title = ""
        content = slide_text
        
        if lines:
            # If first line is short, it's likely a title
            if len(lines[0]) < 80:
                title = lines[0].strip()
                content = '\n'.join(lines[1:]).strip()
                
                # If second line is also short and not a bullet point, add to title
                if len(lines) > 1 and len(lines[1]) < 80 and not lines[1].strip().startswith(('•', '-', '*')):
                    title = f"{title} {lines[1].strip()}"
                    content = '\n'.join(lines[2:]).strip()
        
        return title, content