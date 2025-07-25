# services/processor/src/pipeline/pdf_processor.py
import uuid
import os
import fitz  # PyMuPDF
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

# Keep your existing imports
from services.processor.src.models.document import Document
from services.processor.src.pipeline.content_detector import ContentTypeDetector
from services.processor.src.pipeline.ocr.element_detector import VisualElementDetector
from services.processor.src.pipeline.ocr.tesseract import TesseractOCR

# Add new VLM import
from services.processor.src.pipeline.vlm.vlm_processor import VLMProcessor

class PDFProcessor:
    """Enhanced PDF processor with VLM capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        
        # Initialize existing components
        self.content_detector = ContentTypeDetector()
        self.visual_element_detector = VisualElementDetector()
        self.ocr = TesseractOCR(config.get("ocr", {}))
        
        # Initialize VLM processor
        self.vlm_processor = VLMProcessor(config)
        self.use_vlm = self.vlm_processor.enabled
        
        # Initialize specialized extractors based on config
        self.extractors = {}
        if config.get("enable_document_extractor", True):
            from services.processor.src.pipeline.extractors.document_extractor import DocumentExtractor
            self.extractors["document"] = DocumentExtractor()
            
        if config.get("enable_email_extractor", True):
            from services.processor.src.pipeline.extractors.email_extractor import EmailExtractor
            self.extractors["email"] = EmailExtractor()
            
        if config.get("enable_presentation_extractor", True):
            from services.processor.src.pipeline.extractors.presentation_extractor import PresentationExtractor
            self.extractors["presentation"] = PresentationExtractor()
    
    def process_pdf(self, file_path: str) -> Document:
        """Process a PDF document with optional VLM enhancement."""
        
        if self.use_vlm:
            # Use async VLM processing
            return asyncio.run(self._process_pdf_with_vlm(file_path))
        else:
            # Use traditional processing only
            return self._process_pdf_traditional(file_path)
    
    async def _process_pdf_with_vlm(self, file_path: str) -> Document:
        """Process PDF with VLM enhancement."""
        
        # First, get traditional processing result
        traditional_doc = self._process_pdf_traditional(file_path)
        
        # Then enhance with VLM
        try:
            vlm_result = await self.vlm_processor.process_document(
                file_path, 
                traditional_result=traditional_doc.__dict__
            )
            
            # Merge results
            enhanced_doc = self._merge_traditional_and_vlm(traditional_doc, vlm_result)
            return enhanced_doc
            
        except Exception as e:
            print(f"VLM processing failed, using traditional result: {str(e)}")
            return traditional_doc
    
    def _process_pdf_traditional(self, file_path: str) -> Document:
        """Traditional PDF processing (your existing logic)."""
        
        # Basic document info
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        created_at = datetime.now()
        
        # Extract text and metadata
        doc = fitz.open(file_path)
        all_text = ""
        all_images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            page_text = page.get_text()
            all_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            # Extract images if configured
            if self.config.get("extract_images", False):
                page_images = page.get_images()
                all_images.extend(page_images)
        
        doc.close()
        
        # Detect content type using existing logic
        content_types = self.content_detector.detect_content_type(all_text, file_path)
        
        # Create document
        document = Document(
            doc_id=doc_id,
            file_name=file_name,
            file_path=file_path,
            content_type=content_types,
            text_content=all_text,
            metadata={
                "created_at": created_at.isoformat(),
                "processing_method": "traditional",
                "page_count": len(doc) if 'doc' in locals() else 0,
                "image_count": len(all_images),
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            },
            persuasion_tags=[],  # Keep your existing logic
            embedding=None  # Will be set later
        )
        
        return document
    
    def _merge_traditional_and_vlm(self, traditional_doc: Document, vlm_result: Dict) -> Document:
        """Merge traditional and VLM processing results."""
        
        # Determine best content type
        vlm_content_type = vlm_result.get("document_summary", {}).get("primary_content_type", "mixed")
        traditional_types = traditional_doc.content_type
        
        # Use VLM content type if confidence is high
        vlm_confidence = vlm_result.get("document_summary", {}).get("average_confidence", 0.0)
        
        if vlm_confidence > 0.7:
            final_content_type = [vlm_content_type]
        else:
            # Combine both classifications
            final_content_type = list(set([vlm_content_type] + traditional_types))
        
        # Combine text content
        traditional_text = traditional_doc.text_content
        vlm_text = vlm_result.get("document_summary", {}).get("combined_text_content", "")
        
        # Use VLM text if it's substantially longer (indicates better extraction)
        if len(vlm_text) > len(traditional_text) * 1.2:
            final_text = vlm_text
            text_source = "vlm_enhanced"
        elif len(vlm_text) > 100:  # VLM has reasonable content
            final_text = f"{vlm_text}\n\n--- Traditional Fallback ---\n\n{traditional_text}"
            text_source = "hybrid"
        else:
            final_text = traditional_text
            text_source = "traditional"
        
        # Enhanced metadata
        enhanced_metadata = {
            **traditional_doc.metadata,
            "vlm_processing": {
                "enabled": True,
                "confidence": vlm_confidence,
                "content_type_detected": vlm_content_type,
                "pages_processed": vlm_result.get("document_summary", {}).get("total_pages_processed", 0),
                "successful_pages": vlm_result.get("document_summary", {}).get("successful_vlm_pages", 0),
                "visual_elements": vlm_result.get("document_summary", {}).get("all_visual_elements", []),
                "text_source": text_source
            },
            "processing_method": "vlm_enhanced"
        }
        
        # Create enhanced document
        enhanced_doc = Document(
            doc_id=traditional_doc.doc_id,
            file_name=traditional_doc.file_name,
            file_path=traditional_doc.file_path,
            content_type=final_content_type,
            text_content=final_text,
            metadata=enhanced_metadata,
            persuasion_tags=traditional_doc.persuasion_tags,  # Keep existing
            embedding=traditional_doc.embedding  # Keep existing
        )
        
        return enhanced_doc