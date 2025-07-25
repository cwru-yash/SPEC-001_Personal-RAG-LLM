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

# Import our enhanced components
from pipeline.document_router import DocumentRouter
from pipeline.cloud_processors import GPT4VisionProcessor, ClaudeVisionProcessor
from pipeline.qwen_processor import Qwen25VLProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedVLMDocumentResult:
    """Enhanced document result with multi-model VLM analysis."""
    doc_id: str
    file_name: str
    file_path: str
    category: str  # Documents, Emails, etc.
    doc_type: str  # 'content' or 'metadata'
    content_format: str  # 'pdf', 'xlsx', etc.
    
    # Enhanced VLM Analysis Results
    processing_method: str           # Method used for processing
    primary_model: str              # Primary model that processed the document
    routing_decision: Dict[str, Any] # Information about routing decision
    
    # Content Analysis
    vlm_content_type: str           # email, report, presentation, etc.
    vlm_confidence: float
    vlm_layout_description: str
    vlm_visual_elements: List[Dict]
    
    # Content
    text_content: str
    structured_content: Dict[str, Any]
    
    # Enhanced Metadata
    metadata: Dict[str, Any]
    file_size: int
    processing_time: float
    processing_cost: float = 0.0    # Cost for cloud processing
    
    # Quality metrics
    quality_score: float = 0.0
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class EnhancedDocumentPair:
    """Enhanced document pair with multi-model VLM analysis."""
    pair_id: str
    base_name: str
    category: str
    zip_file: str
    extraction_path: Optional[str] = None
    content_doc: Optional[EnhancedVLMDocumentResult] = None
    metadata_doc: Optional[EnhancedVLMDocumentResult] = None
    pair_analysis: Dict[str, Any] = field(default_factory=dict)
    processing_summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

class EnhancedVLMIntegratedProcessor:
    """Enhanced VLM-first document processor with intelligent multi-model routing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced VLM-integrated processor."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize the document router (our intelligent brain)
        self.document_router = DocumentRouter(config)
        
        # Processing statistics
        self.stats = {
            "zip_files_processed": 0,
            "documents_processed": 0,
            "total_processing_cost": 0.0,
            "model_usage": {},
            "routing_decisions": {},
            "quality_distribution": {},
            "errors": 0,
            "processing_start_time": datetime.now()
        }
        
        # Cost tracking
        self.daily_cost = 0.0
        self.cost_limits = {
            'daily_limit': config.get('vlm', {}).get('monitoring', {}).get('daily_cost_limit', 100.0),
            'document_limit': config.get('vlm', {}).get('monitoring', {}).get('document_cost_limit', 5.0)
        }
        
        # Temporary directory for extractions
        self.temp_dir = tempfile.mkdtemp(prefix="enhanced_vlm_processor_")
        
        self.logger.info(f"Enhanced VLM Integrated Processor initialized")
        self.logger.info(f"Document router available with {len(self.document_router.processors)} processors")
        self.logger.info(f"Cost limits: ${self.cost_limits['document_limit']}/doc, ${self.cost_limits['daily_limit']}/day")
        self.logger.info(f"Temp directory: {self.temp_dir}")
    
    async def process_input_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """Process entire input directory structure with enhanced VLM analysis."""
        
        self.logger.info(f"🚀 Starting enhanced VLM processing of: {input_dir}")
        self.logger.info(f"📁 Output directory: {output_dir}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each category
        category_results = {}
        
        categories = ["Documents", "Emails", "Images", "Presentations", "Spreadsheets"]
        
        for category in categories:
            category_path = os.path.join(input_dir, category)
            
            if os.path.exists(category_path):
                self.logger.info(f"📂 Processing category: {category}")
                
                # Check cost limits before processing category
                if self._check_cost_limits():
                    category_results[category] = await self._process_category(
                        category_path, category, output_dir
                    )
                else:
                    self.logger.warning(f"⚠️ Skipping {category} due to cost limits")
                    category_results[category] = {
                        "status": "skipped",
                        "reason": "cost_limit_exceeded"
                    }
            else:
                self.logger.warning(f"⚠️ Category directory not found: {category_path}")
        
        # Generate comprehensive summary report
        summary = self._generate_enhanced_processing_summary(category_results)
        
        # Save summary to output directory
        summary_path = os.path.join(output_dir, "enhanced_processing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"✅ Enhanced processing completed. Summary saved to: {summary_path}")
        return summary
    
    async def _process_category(self, category_path: str, category: str, output_dir: str) -> Dict[str, Any]:
        """Process all ZIP files in a category with enhanced routing."""
        
        zip_files = [f for f in os.listdir(category_path) if f.endswith('.zip')]
        
        if not zip_files:
            self.logger.warning(f"No ZIP files found in {category_path}")
            return {"zip_files": [], "processed_pairs": []}
        
        self.logger.info(f"Found {len(zip_files)} ZIP files in {category}")
        
        # Create category output directory
        category_output_dir = os.path.join(output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)
        
        processed_pairs = []
        
        # Process ZIP files with enhanced routing
        for zip_file in zip_files:
            zip_path = os.path.join(category_path, zip_file)
            self.logger.info(f"📦 Processing ZIP: {zip_file}")
            
            try:
                # Check cost limits before processing each document
                if not self._check_cost_limits():
                    self.logger.warning(f"Skipping {zip_file} due to cost limits")
                    break
                
                document_pair = await self._process_zip_file_enhanced(zip_path, category, category_output_dir)
                processed_pairs.append(document_pair)
                self.stats["zip_files_processed"] += 1
                
                # Update cost tracking
                pair_cost = (document_pair.content_doc.processing_cost if document_pair.content_doc else 0.0) + \
                           (document_pair.metadata_doc.processing_cost if document_pair.metadata_doc else 0.0)
                self.daily_cost += pair_cost
                self.stats["total_processing_cost"] += pair_cost
                
                # Save individual pair results
                await self._save_enhanced_document_pair_results(document_pair, category_output_dir)
                
            except Exception as e:
                self.logger.error(f"❌ Failed to process {zip_file}: {str(e)}")
                self.stats["errors"] += 1
        
        return {
            "category": category,
            "zip_files": zip_files,
            "processed_pairs": [asdict(pair) for pair in processed_pairs],
            "success_count": len([p for p in processed_pairs if not p.errors]),
            "error_count": len([p for p in processed_pairs if p.errors]),
            "total_cost": sum([
                (p.content_doc.processing_cost if p.content_doc else 0.0) + 
                (p.metadata_doc.processing_cost if p.metadata_doc else 0.0) 
                for p in processed_pairs
            ])
        }
    
    async def _process_zip_file_enhanced(self, zip_path: str, category: str, output_dir: str) -> EnhancedDocumentPair:
        """Process a single ZIP file using enhanced multi-model routing."""
        
        base_name = os.path.splitext(os.path.basename(zip_path))[0]
        pair_id = str(uuid.uuid4())
        
        document_pair = EnhancedDocumentPair(
            pair_id=pair_id,
            base_name=base_name,
            category=category,
            zip_file=zip_path
        )
        
        try:
            # Extract ZIP file
            extraction_path = self._extract_zip_file(zip_path, base_name)
            document_pair.extraction_path = extraction_path
            
            # Find PDF files
            pdf_files = self._find_pdf_files(extraction_path)
            
            if len(pdf_files) < 2:
                raise ValueError(f"Expected 2 PDF files, found {len(pdf_files)}")
            
            # Identify main content and metadata files
            content_pdf, metadata_pdf = self._identify_pdf_pair(pdf_files, base_name)
            
            if not content_pdf or not metadata_pdf:
                raise ValueError("Could not identify content and metadata PDFs")
            
            self.logger.info(f"📄 Content PDF: {os.path.basename(content_pdf)}")
            self.logger.info(f"📄 Metadata PDF: {os.path.basename(metadata_pdf)}")
            
            # Process both files using enhanced routing
            content_result = await self._process_single_document_enhanced(
                content_pdf, category, "content", base_name
            )
            
            metadata_result = await self._process_single_document_enhanced(
                metadata_pdf, category, "metadata", base_name
            )
            
            document_pair.content_doc = content_result
            document_pair.metadata_doc = metadata_result
            
            # Enhanced pair analysis
            document_pair.pair_analysis = self._analyze_enhanced_document_pair(content_result, metadata_result)
            
            # Processing summary
            document_pair.processing_summary = {
                "total_cost": content_result.processing_cost + metadata_result.processing_cost,
                "models_used": [content_result.primary_model, metadata_result.primary_model],
                "routing_efficiency": self._calculate_routing_efficiency(content_result, metadata_result),
                "quality_assessment": self._assess_pair_quality(content_result, metadata_result)
            }
            
            # Update statistics
            self._update_processing_stats(content_result, metadata_result)
            
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
    
    async def _process_single_document_enhanced(self, file_path: str, category: str, 
                                              doc_type: str, base_name: str) -> EnhancedVLMDocumentResult:
        """Process a single document using enhanced multi-model routing."""
        
        start_time = datetime.now()
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        
        self.logger.info(f"🔍 Processing {doc_type}: {file_name} using enhanced routing")
        
        try:
            # Use the document router for intelligent processing
            routing_result = await self.document_router.route_and_process(file_path)
            
            # Extract information from routing result
            if routing_result.get('success', False):
                result_data = routing_result['result']
                processing_method = routing_result['method_used']
                quality_score = routing_result.get('quality_score', 0.0)
                processing_cost = self._calculate_processing_cost(routing_result)
                
                # Extract VLM analysis from result
                if result_data.get('document_summary'):
                    summary = result_data['document_summary']
                    vlm_content_type = summary.get('primary_content_type', 'mixed')
                    vlm_confidence = summary.get('average_confidence', 0.0)
                    combined_text = summary.get('combined_text_content', '')
                    visual_elements = summary.get('all_visual_elements', [])
                    layout_description = f"Processed with {processing_method}"
                else:
                    # Fallback processing results
                    vlm_content_type = 'mixed'
                    vlm_confidence = 0.5
                    combined_text = routing_result.get('extracted_text', '')
                    visual_elements = []
                    layout_description = f"Fallback processing with {processing_method}"
                
            else:
                # Processing failed completely
                processing_method = routing_result.get('method_used', 'failed')
                quality_score = 0.0
                processing_cost = 0.0
                vlm_content_type = 'error'
                vlm_confidence = 0.0
                combined_text = f"Processing failed: {routing_result.get('error', 'Unknown error')}"
                visual_elements = []
                layout_description = "Processing failed"
            
            # Create structured content
            structured_content = self._create_enhanced_structured_content(
                combined_text, vlm_content_type, doc_type, visual_elements, routing_result
            )
            
            # Enhanced metadata with routing information
            routing_info = routing_result.get('routing_info', {})
            metadata = {
                "pair_id": base_name,
                "category": category,
                "doc_type": doc_type,
                "processing_timestamp": datetime.now().isoformat(),
                "file_stats": {
                    "size_bytes": os.path.getsize(file_path),
                    "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2)
                },
                "enhanced_vlm_processing": {
                    "routing_enabled": True,
                    "method_used": processing_method,
                    "confidence": vlm_confidence,
                    "content_type_detected": vlm_content_type,
                    "quality_score": quality_score,
                    "processing_cost": processing_cost
                },
                "routing_information": {
                    "document_characteristics": asdict(routing_info.get('characteristics', {})) if routing_info.get('characteristics') else {},
                    "processing_strategy": asdict(routing_info.get('strategy', {})) if routing_info.get('strategy') else {},
                    "routing_decision_confidence": routing_info.get('characteristics', {}).confidence if routing_info.get('characteristics') else 0.0
                }
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnhancedVLMDocumentResult(
                doc_id=doc_id,
                file_name=file_name,
                file_path=file_path,
                category=category,
                doc_type=doc_type,
                content_format="pdf",
                processing_method=processing_method,
                primary_model=self._extract_primary_model(processing_method),
                routing_decision=routing_info,
                vlm_content_type=vlm_content_type,
                vlm_confidence=vlm_confidence,
                vlm_layout_description=layout_description,
                vlm_visual_elements=visual_elements,
                text_content=combined_text,
                structured_content=structured_content,
                metadata=metadata,
                file_size=os.path.getsize(file_path),
                processing_time=processing_time,
                processing_cost=processing_cost,
                quality_score=quality_score,
                success=routing_result.get('success', False)
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Failed to process {file_name}: {str(e)}")
            
            return EnhancedVLMDocumentResult(
                doc_id=doc_id,
                file_name=file_name,
                file_path=file_path,
                category=category,
                doc_type=doc_type,
                content_format="pdf",
                processing_method="failed",
                primary_model="none",
                routing_decision={})