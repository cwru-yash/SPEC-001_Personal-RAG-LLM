# src/pipeline/pdf_processor.py
import uuid
import os
import fitz  # PyMuPDF
from typing import Dict, List, Optional, Any
from datetime import datetime

# from src.models.document import Document
from services.processor.src.models.document import Document
# from src.pipeline.content_detector import ContentTypeDetector
from services.processor.src.pipeline.content_detector import ContentTypeDetector
# from src.pipeline.ocr.element_detector import VisualElementDetector
# from src.pipeline.ocr.tesseract import TesseractOCR
from services.processor.src.pipeline.ocr.element_detector import VisualElementDetector
from services.processor.src.pipeline.ocr.tesseract import TesseractOCR



class PDFProcessor:
    """Main processor for PDF documents with mixed content types."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.content_detector = ContentTypeDetector()
        self.visual_element_detector = VisualElementDetector()
        self.ocr = TesseractOCR(config.get("ocr", {}))
        
        # Initialize specialized extractors based on config
        self.extractors = {}
        if config.get("enable_document_extractor", True):
            from src.pipeline.extractors.document_extractor import DocumentExtractor
            self.extractors["document"] = DocumentExtractor()
            
        if config.get("enable_email_extractor", True):
            from src.pipeline.extractors.email_extractor import EmailExtractor
            self.extractors["email"] = EmailExtractor()
            
        if config.get("enable_presentation_extractor", True):
            from src.pipeline.extractors.presentation_extractor import PresentationExtractor
            self.extractors["presentation"] = PresentationExtractor()
    
    def process_pdf(self, file_path: str) -> Document:
        """Process a PDF document with content type detection and specialized extraction."""
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        
        base_document = Document(
            doc_id=doc_id,
            file_name=file_name,
            file_extension="pdf",
            content_type=["pdf"],  # Will be enhanced later
            created_at=None,  # Will be extracted from metadata
            text_content="",
            metadata={}
        )
        
        try:
            # Open PDF with PyMuPDF
            with fitz.open(file_path) as pdf:
                # Extract basic metadata
                base_document.metadata = {
                    "page_count": len(pdf),
                    "title": pdf.metadata.get("title", ""),
                    "author": pdf.metadata.get("author", ""),
                    "creation_date": pdf.metadata.get("creationDate", ""),
                    "modification_date": pdf.metadata.get("modDate", ""),
                }
                
                # Set author and created_at if available
                base_document.author = pdf.metadata.get("author", "")
                if pdf.metadata.get("creationDate"):
                    # PyMuPDF dates need special parsing - this is simplified
                    try:
                        # Convert PDF date format to datetime
                        date_str = pdf.metadata.get("creationDate")
                        # Remove D: prefix and Z suffix if present
                        if date_str.startswith("D:"):
                            date_str = date_str[2:]
                        if date_str.endswith("Z"):
                            date_str = date_str[:-1]
                        # Format: YYYYMMDDHHmmSS
                        base_document.created_at = datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
                    except:
                        pass
                
                # Detect content type
                sample_pages = self._get_sample_pages(pdf)
                content_types = self.content_detector.detect_content_types(sample_pages)
                base_document.content_type.extend(content_types)
                
                # Use specialized extractor based on content type
                primary_content_type = self._get_primary_content_type(content_types)
                
                if primary_content_type in self.extractors:
                    # Use specialized extractor
                    extractor = self.extractors[primary_content_type]
                    extracted_doc = extractor.extract(pdf, base_document)
                else:
                    # Fallback to default extraction
                    extracted_doc = self._default_extraction(pdf, base_document)
                
                # Process visual elements with OCR if needed
                if self.config.get("enable_ocr", True):
                    extracted_doc = self._process_visual_elements(pdf, extracted_doc)
                    
                return extracted_doc
                
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return base_document
    
    def _get_sample_pages(self, pdf):
        """Get a sample of pages for content type detection."""
        # Get first, middle and last page if available
        page_count = len(pdf)
        sample_pages = []
        
        if page_count > 0:
            sample_pages.append(pdf[0])
        
        if page_count > 2:
            sample_pages.append(pdf[page_count // 2])
        
        if page_count > 1:
            sample_pages.append(pdf[page_count - 1])
            
        return sample_pages
    
    def _get_primary_content_type(self, content_types):
        """Determine the primary content type from detected types."""
        # Priority order: email > presentation > document (default)
        if "email" in content_types:
            return "email"
        elif "presentation" in content_types:
            return "presentation"
        else:
            return "document"
    
    def _default_extraction(self, pdf, base_document):
        """Default text extraction when no specialized extractor is available."""
        text_content = ""
        
        # Extract text from all pages
        for page in pdf:
            text_content += page.get_text()
            
        base_document.text_content = text_content
        return base_document
    
    def _process_visual_elements(self, pdf, document):
        """Detect and process visual elements using OCR."""
        # Track if document has charts or images
        has_charts = False
        has_images = False
        visual_element_text = ""
        
        # Check each page for visual elements
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            
            # Detect images and charts
            images = self.visual_element_detector.detect_images(page)
            charts = self.visual_element_detector.detect_charts(page)
            
            if images:
                has_images = True
                
            if charts:
                has_charts = True
                
            # Process each visual element with OCR
            for element in images + charts:
                # Extract image to temporary file
                element_image = element.extract_image()
                if element_image:
                    ocr_text = self.ocr.extract_text(element_image)
                    if ocr_text:
                        visual_element_text += f"\n[Visual Element - Page {page_num+1}]: {ocr_text}\n"
        
        # Update document metadata and content
        document.metadata["has_charts"] = has_charts
        document.metadata["has_images"] = has_images
        
        # Append OCR text to document content
        if visual_element_text:
            document.text_content += "\n\n--- Visual Elements (OCR) ---\n" + visual_element_text
            document.metadata["ocr_applied"] = True
            
        return document