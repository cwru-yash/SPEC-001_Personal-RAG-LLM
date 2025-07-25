# services/processor/src/pipeline/vlm_integrated_processor.py

import os
import sys
import zipfile
import tempfile
import shutil
import json
import uuid
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
import fitz  # PyMuPDF

# Import our VLM processor
# from services.processor.src.pipeline.vlm.cpu_optimized_processor import CPUOptimizedVLMProcessor
from pipeline.vlm.cpu_optimized_processor import CPUOptimizedVLMProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VLMDocumentResult:
    """Enhanced document result with VLM analysis."""
    doc_id: str
    file_name: str
    file_path: str
    category: str  # Documents, Emails, etc.
    doc_type: str  # 'content' or 'metadata'
    content_format: str  # 'pdf', 'xlsx', etc.
    
    # VLM Analysis Results
    vlm_content_type: str  # email, report, presentation, etc.
    vlm_confidence: float
    vlm_layout_description: str
    vlm_visual_elements: List[Dict]
    
    # Content
    text_content: str
    structured_content: Dict[str, Any]
    
    # Metadata
    metadata: Dict[str, Any]
    file_size: int
    processing_time: float
    processing_method: str  # "vlm_enhanced", "vlm_only", "traditional_fallback"
    
    # Quality metrics
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class DocumentPair:
    """Document pair with VLM analysis."""
    pair_id: str
    base_name: str
    category: str
    zip_file: str
    extraction_path: Optional[str] = None
    content_doc: Optional[VLMDocumentResult] = None
    metadata_doc: Optional[VLMDocumentResult] = None
    pair_analysis: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

class VLMIntegratedProcessor:
    """VLM-first document processor with ZIP support."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the VLM-integrated processor."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize VLM processor
        self.vlm_processor = CPUOptimizedVLMProcessor(config)
        
        # Processing statistics
        self.stats = {
            "zip_files_processed": 0,
            "documents_processed": 0,
            "vlm_successes": 0,
            "vlm_fallbacks": 0,
            "errors": 0,
            "processing_start_time": datetime.now()
        }
        
        # Temporary directory for extractions
        self.temp_dir = tempfile.mkdtemp(prefix="vlm_processor_")
        self.logger.info(f"VLM Integrated Processor initialized")
        self.logger.info(f"VLM enabled: {self.vlm_processor.enabled}")
        self.logger.info(f"Temp directory: {self.temp_dir}")
    
    async def process_input_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """Process entire input directory structure with VLM analysis."""
        
        self.logger.info(f"🚀 Starting VLM-integrated processing of: {input_dir}")
        self.logger.info(f"📁 Output directory: {output_dir}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all categories and ZIP files
        category_results = {}
        
        for category in ["Documents", "Emails", "Images", "Presentations", "Spreadsheets"]:
            category_path = os.path.join(input_dir, category)
            
            if os.path.exists(category_path):
                self.logger.info(f"📂 Processing category: {category}")
                category_results[category] = await self._process_category(
                    category_path, category, output_dir
                )
            else:
                self.logger.warning(f"⚠️ Category directory not found: {category_path}")
        
        # Generate summary report
        summary = self._generate_processing_summary(category_results)
        
        # Save summary to output directory
        summary_path = os.path.join(output_dir, "processing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"✅ Processing completed. Summary saved to: {summary_path}")
        return summary
    
    async def _process_category(self, category_path: str, category: str, output_dir: str) -> Dict[str, Any]:
        """Process all ZIP files in a category."""
        
        zip_files = [f for f in os.listdir(category_path) if f.endswith('.zip')]
        
        if not zip_files:
            self.logger.warning(f"No ZIP files found in {category_path}")
            return {"zip_files": [], "processed_pairs": []}
        
        self.logger.info(f"Found {len(zip_files)} ZIP files in {category}")
        
        # Create category output directory
        category_output_dir = os.path.join(output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)
        
        processed_pairs = []
        
        # Process ZIP files (can be parallelized later if needed)
        for zip_file in zip_files:
            zip_path = os.path.join(category_path, zip_file)
            self.logger.info(f"📦 Processing ZIP: {zip_file}")
            
            try:
                document_pair = await self._process_zip_file(zip_path, category, category_output_dir)
                processed_pairs.append(document_pair)
                self.stats["zip_files_processed"] += 1
                
                # Save individual pair results
                await self._save_document_pair_results(document_pair, category_output_dir)
                
            except Exception as e:
                self.logger.error(f"❌ Failed to process {zip_file}: {str(e)}")
                self.stats["errors"] += 1
        
        return {
            "category": category,
            "zip_files": zip_files,
            "processed_pairs": [asdict(pair) for pair in processed_pairs],
            "success_count": len([p for p in processed_pairs if not p.errors]),
            "error_count": len([p for p in processed_pairs if p.errors])
        }
    
    async def _process_zip_file(self, zip_path: str, category: str, output_dir: str) -> DocumentPair:
        """Process a single ZIP file containing document pair."""
        
        base_name = os.path.splitext(os.path.basename(zip_path))[0]
        pair_id = str(uuid.uuid4())
        
        document_pair = DocumentPair(
            pair_id=pair_id,
            base_name=base_name,
            category=category,
            zip_file=zip_path
        )
        
        try:
            # Extract ZIP file
            extraction_path = self._extract_zip_file(zip_path, base_name)
            document_pair.extraction_path = extraction_path
            
            # Find PDF files (assuming they're PDFs based on your description)
            pdf_files = self._find_pdf_files(extraction_path)
            
            if len(pdf_files) < 2:
                raise ValueError(f"Expected 2 PDF files, found {len(pdf_files)}")
            
            # Identify main content and metadata files
            content_pdf, metadata_pdf = self._identify_pdf_pair(pdf_files, base_name)
            
            if not content_pdf or not metadata_pdf:
                raise ValueError("Could not identify content and metadata PDFs")
            
            self.logger.info(f"📄 Content PDF: {os.path.basename(content_pdf)}")
            self.logger.info(f"📄 Metadata PDF: {os.path.basename(metadata_pdf)}")
            
            # Process both files with VLM
            content_result = await self._process_single_document(
                content_pdf, category, "content", base_name
            )
            
            metadata_result = await self._process_single_document(
                metadata_pdf, category, "metadata", base_name
            )
            
            document_pair.content_doc = content_result
            document_pair.metadata_doc = metadata_result
            
            # Analyze the pair relationship
            document_pair.pair_analysis = self._analyze_document_pair(content_result, metadata_result)
            
            # Update statistics
            if content_result.success:
                self.stats["documents_processed"] += 1
                if content_result.processing_method.startswith("vlm"):
                    self.stats["vlm_successes"] += 1
                else:
                    self.stats["vlm_fallbacks"] += 1
            
            if metadata_result.success:
                self.stats["documents_processed"] += 1
                if metadata_result.processing_method.startswith("vlm"):
                    self.stats["vlm_successes"] += 1
                else:
                    self.stats["vlm_fallbacks"] += 1
            
        except Exception as e:
            document_pair.errors.append(str(e))
            self.logger.error(f"Error processing {base_name}: {str(e)}")
            self.stats["errors"] += 1
        
        finally:
            # Cleanup extraction directory
            if document_pair.extraction_path and os.path.exists(document_pair.extraction_path):
                try:
                    shutil.rmtree(document_pair.extraction_path)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup {document_pair.extraction_path}: {e}")
        
        return document_pair
    
    async def _process_single_document(self, file_path: str, category: str, 
                                     doc_type: str, base_name: str) -> VLMDocumentResult:
        """Process a single document with VLM as primary method."""
        
        start_time = datetime.now()
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        
        self.logger.info(f"🔍 Processing {doc_type}: {file_name}")
        
        try:
            # Use VLM as primary processing method
            if self.vlm_processor.enabled:
                vlm_result = await self.vlm_processor.process_document_with_progress(file_path)
                
                # Extract VLM analysis
                if vlm_result.get("document_summary"):
                    summary = vlm_result["document_summary"]
                    processing_method = "vlm_enhanced"
                    
                    # Get the best content from VLM analysis
                    vlm_content = summary.get("combined_text_content", "")
                    vlm_content_type = summary.get("primary_content_type", "mixed")
                    vlm_confidence = summary.get("average_confidence", 0.0)
                    
                    # Extract layout and visual elements
                    pages = vlm_result.get("pages", [])
                    layout_descriptions = []
                    visual_elements = []
                    
                    for page in pages:
                        vlm_page_result = page.get("vlm_result", {})
                        if vlm_page_result.get("success"):
                            layout_descriptions.append(vlm_page_result.get("layout_description", ""))
                            visual_elements.extend(vlm_page_result.get("visual_elements", []))
                    
                    combined_layout = " | ".join(filter(None, layout_descriptions))
                    
                else:
                    # VLM failed, use traditional fallback
                    self.logger.warning(f"VLM processing failed for {file_name}, using traditional fallback")
                    processing_method = "traditional_fallback"
                    vlm_content = self._extract_traditional_text(file_path)
                    vlm_content_type = "mixed"
                    vlm_confidence = 0.0
                    combined_layout = "Traditional extraction - no layout analysis"
                    visual_elements = []
            else:
                # VLM disabled, use traditional processing
                processing_method = "traditional_only"
                vlm_content = self._extract_traditional_text(file_path)
                vlm_content_type = "mixed"
                vlm_confidence = 0.0
                combined_layout = "VLM disabled - traditional extraction only"
                visual_elements = []
            
            # Create structured content based on document type
            structured_content = self._create_structured_content(
                vlm_content, vlm_content_type, doc_type, visual_elements
            )
            
            # Enhanced metadata
            metadata = {
                "pair_id": base_name,
                "category": category,
                "doc_type": doc_type,
                "processing_timestamp": datetime.now().isoformat(),
                "file_stats": {
                    "size_bytes": os.path.getsize(file_path),
                    "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2)
                },
                "vlm_processing": {
                    "enabled": self.vlm_processor.enabled,
                    "method": processing_method,
                    "confidence": vlm_confidence,
                    "content_type_detected": vlm_content_type
                }
            }
            
            # Add VLM-specific metadata if available
            if 'vlm_result' in locals() and vlm_result.get("performance_metrics"):
                metadata["vlm_processing"]["performance"] = vlm_result["performance_metrics"]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return VLMDocumentResult(
                doc_id=doc_id,
                file_name=file_name,
                file_path=file_path,
                category=category,
                doc_type=doc_type,
                content_format="pdf",
                vlm_content_type=vlm_content_type,
                vlm_confidence=vlm_confidence,
                vlm_layout_description=combined_layout,
                vlm_visual_elements=visual_elements,
                text_content=vlm_content,
                structured_content=structured_content,
                metadata=metadata,
                file_size=os.path.getsize(file_path),
                processing_time=processing_time,
                processing_method=processing_method,
                success=True
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Failed to process {file_name}: {str(e)}")
            
            return VLMDocumentResult(
                doc_id=doc_id,
                file_name=file_name,
                file_path=file_path,
                category=category,
                doc_type=doc_type,
                content_format="pdf",
                vlm_content_type="error",
                vlm_confidence=0.0,
                vlm_layout_description="",
                vlm_visual_elements=[],
                text_content="",
                structured_content={},
                metadata={"error": str(e)},
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                processing_time=processing_time,
                processing_method="failed",
                success=False,
                errors=[str(e)]
            )
    
    def _extract_traditional_text(self, file_path: str) -> str:
        """Fallback traditional text extraction."""
        try:
            doc = fitz.open(file_path)
            text_content = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_content += f"\n--- Page {page_num + 1} ---\n"
                text_content += page.get_text()
            
            doc.close()
            return text_content
        except Exception as e:
            self.logger.error(f"Traditional text extraction failed: {e}")
            return f"[Text extraction failed: {str(e)}]"
    
    def _create_structured_content(self, text_content: str, content_type: str, 
                                 doc_type: str, visual_elements: List[Dict]) -> Dict[str, Any]:
        """Create structured content based on analysis."""
        
        structured = {
            "content_type": content_type,
            "document_role": doc_type,
            "text_length": len(text_content),
            "has_visual_elements": len(visual_elements) > 0,
            "visual_element_count": len(visual_elements),
            "visual_elements": visual_elements
        }
        
        # Add content-type specific structure
        if content_type == "email":
            structured.update(self._extract_email_structure(text_content))
        elif content_type == "report":
            structured.update(self._extract_report_structure(text_content))
        elif content_type == "presentation":
            structured.update(self._extract_presentation_structure(text_content))
        
        return structured
    
    def _extract_email_structure(self, text: str) -> Dict[str, Any]:
        """Extract email-specific structure."""
        # Simple email parsing - can be enhanced
        lines = text.split('\n')
        email_data = {
            "email_structure": True,
            "potential_header_lines": [line for line in lines[:10] if ':' in line],
            "estimated_header_length": len([line for line in lines[:20] if ':' in line])
        }
        return email_data
    
    def _extract_report_structure(self, text: str) -> Dict[str, Any]:
        """Extract report-specific structure."""
        return {
            "report_structure": True,
            "estimated_sections": len([line for line in text.split('\n') if line.strip() and len(line.strip()) < 100])
        }
    
    def _extract_presentation_structure(self, text: str) -> Dict[str, Any]:
        """Extract presentation-specific structure."""
        return {
            "presentation_structure": True,
            "estimated_slides": text.count('---') + 1  # Simple slide estimation
        }
    
    def _extract_zip_file(self, zip_path: str, base_name: str) -> str:
        """Extract ZIP file to temporary directory."""
        extraction_path = os.path.join(self.temp_dir, f"{base_name}_{uuid.uuid4().hex[:8]}")
        os.makedirs(extraction_path, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
        
        return extraction_path
    
    def _find_pdf_files(self, extraction_path: str) -> List[str]:
        """Find all PDF files in extracted directory."""
        pdf_files = []
        for root, _, files in os.walk(extraction_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files
    
    def _identify_pdf_pair(self, pdf_files: List[str], base_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Identify content vs metadata PDF files."""
        metadata_pdf = None
        content_pdf = None
        
        # Look for metadata patterns
        for pdf_file in pdf_files:
            file_name = os.path.basename(pdf_file)
            name_without_ext = os.path.splitext(file_name)[0]
            
            if (name_without_ext.endswith('-info') or 
                'info' in name_without_ext.lower() or
                'metadata' in name_without_ext.lower()):
                metadata_pdf = pdf_file
            elif name_without_ext == base_name or name_without_ext.startswith(base_name):
                content_pdf = pdf_file
        
        # Fallback: use file size (metadata usually smaller)
        if not metadata_pdf or not content_pdf:
            pdf_files_with_size = [(f, os.path.getsize(f)) for f in pdf_files]
            pdf_files_with_size.sort(key=lambda x: x[1])
            
            metadata_pdf = pdf_files_with_size[0][0]
            content_pdf = pdf_files_with_size[1][0]
        
        return content_pdf, metadata_pdf
    
    def _analyze_document_pair(self, content_doc: VLMDocumentResult, 
                             metadata_doc: VLMDocumentResult) -> Dict[str, Any]:
        """Analyze relationship between document pair."""
        
        if not content_doc or not metadata_doc:
            return {"error": "Missing document in pair"}
        
        analysis = {
            "content_type_match": content_doc.vlm_content_type == metadata_doc.vlm_content_type,
            "confidence_comparison": {
                "content_confidence": content_doc.vlm_confidence,
                "metadata_confidence": metadata_doc.vlm_confidence,
                "average_confidence": (content_doc.vlm_confidence + metadata_doc.vlm_confidence) / 2
            },
            "size_comparison": {
                "content_size": content_doc.file_size,
                "metadata_size": metadata_doc.file_size,
                "size_ratio": content_doc.file_size / metadata_doc.file_size if metadata_doc.file_size > 0 else 0
            },
            "processing_comparison": {
                "content_method": content_doc.processing_method,
                "metadata_method": metadata_doc.processing_method,
                "both_vlm_success": (content_doc.processing_method.startswith("vlm") and 
                                   metadata_doc.processing_method.startswith("vlm"))
            }
        }
        
        return analysis
    
    async def _save_document_pair_results(self, document_pair: DocumentPair, output_dir: str):
        """Save document pair results as JSON files."""
        
        pair_dir = os.path.join(output_dir, document_pair.base_name)
        os.makedirs(pair_dir, exist_ok=True)
        
        # Save content document result
        if document_pair.content_doc:
            content_path = os.path.join(pair_dir, f"{document_pair.base_name}_content.json")
            with open(content_path, 'w') as f:
                json.dump(asdict(document_pair.content_doc), f, indent=2, default=str)
        
        # Save metadata document result
        if document_pair.metadata_doc:
            metadata_path = os.path.join(pair_dir, f"{document_pair.base_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(asdict(document_pair.metadata_doc), f, indent=2, default=str)
        
        # Save pair analysis
        pair_analysis_path = os.path.join(pair_dir, f"{document_pair.base_name}_pair_analysis.json")
        pair_summary = {
            "pair_id": document_pair.pair_id,
            "base_name": document_pair.base_name,
            "category": document_pair.category,
            "zip_file": document_pair.zip_file,
            "pair_analysis": document_pair.pair_analysis,
            "errors": document_pair.errors,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        with open(pair_analysis_path, 'w') as f:
            json.dump(pair_summary, f, indent=2, default=str)
        
        self.logger.info(f"💾 Saved results for {document_pair.base_name} to {pair_dir}")
    
    def _generate_processing_summary(self, category_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive processing summary."""
        
        total_processing_time = (datetime.now() - self.stats["processing_start_time"]).total_seconds()
        
        summary = {
            "processing_summary": {
                "start_time": self.stats["processing_start_time"].isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_processing_time_seconds": total_processing_time,
                "total_processing_time_formatted": f"{total_processing_time/60:.1f} minutes"
            },
            "statistics": {
                "zip_files_processed": self.stats["zip_files_processed"],
                "documents_processed": self.stats["documents_processed"],
                "vlm_successes": self.stats["vlm_successes"],
                "vlm_fallbacks": self.stats["vlm_fallbacks"],
                "errors": self.stats["errors"],
                "vlm_success_rate": (self.stats["vlm_successes"] / self.stats["documents_processed"] 
                                   if self.stats["documents_processed"] > 0 else 0)
            },
            "category_results": category_results,
            "vlm_processor_info": {
                "enabled": self.vlm_processor.enabled,
                "model": getattr(self.vlm_processor, 'model', 'unknown'),
                "performance_metrics": asdict(self.vlm_processor.performance_metrics) if hasattr(self.vlm_processor, 'performance_metrics') else {}
            }
        }
        
        return summary
    
    def cleanup(self):
        """Cleanup temporary directories and resources."""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temporary directory: {e}")

    def __del__(self):
        """Ensure cleanup on destruction."""
        self.cleanup()