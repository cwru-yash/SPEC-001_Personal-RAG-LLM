# services/processor/src/pipeline/document_pipeline.py
import os
import uuid
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# from src.models.document import Document
# from src.pipeline.content_detector import ContentTypeDetector
# from src.pipeline.pdf_processor import PDFProcessor
# from src.pipeline.chunkers.text_chunker import ContentAwareChunker
# from src.pipeline.classifier import DocumentClassifier

# Fix the imports to use relative paths
from ..models.document import Document
from .content_detector import ContentTypeDetector
from .pdf_processor import PDFProcessor
from .chunkers.text_chunker import ContentAwareChunker
from .classifier import DocumentClassifier

@dataclass
class ProcessingResult:
    """Result of document processing operation."""
    success: bool
    document: Optional[Document] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None

class DocumentProcessor:
    """Registry for document processors by file type."""
    
    def __init__(self):
        self._processors = {}
        self._validators = {}
    
    def register_processor(self, file_extension: str, processor: Callable):
        """Register a processor for a specific file extension."""
        self._processors[file_extension.lower()] = processor
        logging.info(f"Registered processor for {file_extension} files")
    
    def register_validator(self, file_extension: str, validator: Callable):
        """Register a validator for a specific file extension."""
        self._validators[file_extension.lower()] = validator
    
    def get_processor(self, file_extension: str) -> Optional[Callable]:
        """Get processor for file extension."""
        return self._processors.get(file_extension.lower())
    
    def get_validator(self, file_extension: str) -> Optional[Callable]:
        """Get validator for file extension."""
        return self._validators.get(file_extension.lower())
    
    def supported_types(self) -> List[str]:
        """Get list of supported file types."""
        return list(self._processors.keys())

class DocumentProcessingPipeline:
    """Main document processing pipeline with modular architecture."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the processing pipeline."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.content_detector = ContentTypeDetector()
        self.chunker = ContentAwareChunker(config.get("chunker", {}))
        self.classifier = DocumentClassifier(config.get("classifier", {}))
        
        # Initialize processor registry
        self.processor_registry = DocumentProcessor()
        self._setup_default_processors()
        
        # Processing statistics
        self.stats = {
            "processed": 0,
            "failed": 0,
            "total_time": 0.0,
            "by_type": {}
        }
    
    def _setup_default_processors(self):
        """Setup default processors for common file types."""
        # PDF processor
        pdf_config = self.config.get("pdf", {})
        pdf_processor = PDFProcessor(pdf_config)
        self.processor_registry.register_processor("pdf", pdf_processor.process_pdf)

            
    # Add new extractors
        from src.pipeline.extractors.office_extractor import OfficeExtractor
        from src.pipeline.extractors.image_extractor import ImageExtractor
        
        # Office documents
        office_extractor = OfficeExtractor(self.config.get("office", {}))
        for ext in ["xlsx", "xls", "docx", "doc", "pptx", "ppt"]:
            self.processor_registry.register_processor(ext, office_extractor.extract)
        
        # Images
        image_extractor = ImageExtractor(self.config.get("image", {}))
        for ext in ["png", "jpg", "jpeg", "tiff", "bmp"]:
            self.processor_registry.register_processor(ext, image_extractor.extract)
        
        # Add validators
        self.processor_registry.register_validator("pdf", self._validate_pdf)
        
        # Register additional processors
        self._register_office_processors()
        self._register_image_processors()
        self._register_text_processors()
    
    def _register_office_processors(self):
        """Register processors for Office documents."""
        try:
            from src.pipeline.extractors.office_extractor import OfficeExtractor
            office_extractor = OfficeExtractor(self.config.get("office", {}))
            
            # Register for different Office formats
            for ext in ["docx", "doc", "pptx", "ppt", "xlsx", "xls"]:
                self.processor_registry.register_processor(ext, office_extractor.extract)
                self.processor_registry.register_validator(ext, self._validate_office)
                
        except ImportError:
            self.logger.warning("Office extractor not available - install python-docx, python-pptx, openpyxl")
    
    def _register_image_processors(self):
        """Register processors for image files."""
        try:
            from src.pipeline.extractors.image_extractor import ImageExtractor
            image_extractor = ImageExtractor(self.config.get("image", {}))
            
            for ext in ["png", "jpg", "jpeg", "tiff", "bmp"]:
                self.processor_registry.register_processor(ext, image_extractor.extract)
                self.processor_registry.register_validator(ext, self._validate_image)
                
        except ImportError:
            self.logger.warning("Image extractor not available - install Pillow")
    
    def _register_text_processors(self):
        """Register processors for text files."""
        from src.pipeline.extractors.text_extractor import TextExtractor
        text_extractor = TextExtractor(self.config.get("text", {}))
        
        for ext in ["txt", "md", "csv", "json", "xml", "html"]:
            self.processor_registry.register_processor(ext, text_extractor.extract)
            self.processor_registry.register_validator(ext, self._validate_text)
    
    def register_custom_processor(self, file_extension: str, processor: Callable, validator: Optional[Callable] = None):
        """Register a custom processor for specific file types."""
        self.processor_registry.register_processor(file_extension, processor)
        if validator:
            self.processor_registry.register_validator(file_extension, validator)
        
        self.logger.info(f"Registered custom processor for {file_extension} files")
    
    def process_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process a single file through the pipeline."""
        start_time = datetime.now()
        
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                return ProcessingResult(
                    success=False,
                    error=f"File not found: {file_path}"
                )
            
            # Get file info
            file_path_obj = Path(file_path)
            file_extension = file_path_obj.suffix[1:].lower()  # Remove the dot
            file_size = os.path.getsize(file_path)
            
            # Check if we support this file type
            processor = self.processor_registry.get_processor(file_extension)
            if not processor:
                return ProcessingResult(
                    success=False,
                    error=f"Unsupported file type: {file_extension}"
                )
            
            # Validate file
            validator = self.processor_registry.get_validator(file_extension)
            if validator and not validator(file_path):
                return ProcessingResult(
                    success=False,
                    error=f"File validation failed: {file_path}"
                )
            
            # Check file size limits
            max_file_size = self.config.get("max_file_size_mb", 100) * 1024 * 1024
            if file_size > max_file_size:
                return ProcessingResult(
                    success=False,
                    error=f"File too large: {file_size} bytes (max: {max_file_size})"
                )
            
            # Process the document
            document = processor(file_path)
            
            # Add additional metadata
            if metadata:
                document.metadata.update(metadata)
            
            # Add processing metadata
            document.metadata.update({
                "file_size": file_size,
                "processed_at": datetime.now().isoformat(),
                "processor_version": "1.0",
                "file_extension": file_extension
            })
            
            # Classify document (add tags and persuasion strategies)
            document = self.classifier.classify(document)
            
            # Chunk document (content-aware)
            document = self.chunker.chunk_document(document)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self._update_stats(file_extension, processing_time, success=True)
            
            return ProcessingResult(
                success=True,
                document=document,
                processing_time=processing_time,
                metadata={
                    "chunks_created": len(document.chunks),
                    "content_types": document.content_type,
                    "file_size": file_size
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(file_extension, processing_time, success=False)
            
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def process_directory(self, directory_path: str, recursive: bool = True, max_workers: int = 4) -> List[ProcessingResult]:
        """Process all supported files in a directory."""
        results = []
        files_to_process = []
        
        # Collect all files to process
        supported_extensions = set(self.processor_registry.supported_types())
        
        if recursive:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = Path(file).suffix[1:].lower()
                    if file_ext in supported_extensions:
                        files_to_process.append(file_path)
        else:
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    file_ext = Path(file).suffix[1:].lower()
                    if file_ext in supported_extensions:
                        files_to_process.append(file_path)
        
        self.logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, file_path): file_path 
                for file_path in files_to_process
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        self.logger.info(f"Successfully processed: {file_path}")
                    else:
                        self.logger.error(f"Failed to process {file_path}: {result.error}")
                        
                except Exception as e:
                    self.logger.error(f"Exception processing {file_path}: {str(e)}")
                    results.append(ProcessingResult(
                        success=False,
                        error=str(e)
                    ))
        
        return results
    
    def process_batch(self, file_paths: List[str], max_workers: int = 4) -> List[ProcessingResult]:
        """Process a batch of files."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, file_path): file_path 
                for file_path in file_paths
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Exception processing {file_path}: {str(e)}")
                    results.append(ProcessingResult(
                        success=False,
                        error=str(e)
                    ))
        
        return results
    
    def _update_stats(self, file_extension: str, processing_time: float, success: bool):
        """Update processing statistics."""
        if success:
            self.stats["processed"] += 1
        else:
            self.stats["failed"] += 1
        
        self.stats["total_time"] += processing_time
        
        if file_extension not in self.stats["by_type"]:
            self.stats["by_type"][file_extension] = {
                "processed": 0,
                "failed": 0,
                "total_time": 0.0
            }
        
        if success:
            self.stats["by_type"][file_extension]["processed"] += 1
        else:
            self.stats["by_type"][file_extension]["failed"] += 1
        
        self.stats["by_type"][file_extension]["total_time"] += processing_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_files = self.stats["processed"] + self.stats["failed"]
        avg_time = self.stats["total_time"] / total_files if total_files > 0 else 0
        
        return {
            "total_files": total_files,
            "successful": self.stats["processed"],
            "failed": self.stats["failed"],
            "success_rate": self.stats["processed"] / total_files if total_files > 0 else 0,
            "average_processing_time": avg_time,
            "total_processing_time": self.stats["total_time"],
            "by_file_type": self.stats["by_type"],
            "supported_types": self.processor_registry.supported_types()
        }
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            "processed": 0,
            "failed": 0,
            "total_time": 0.0,
            "by_type": {}
        }
    
    # Validation methods
    def _validate_pdf(self, file_path: str) -> bool:
        """Validate PDF file."""
        try:
            import fitz
            doc = fitz.open(file_path)
            doc.close()
            return True
        except:
            return False
    
    def _validate_office(self, file_path: str) -> bool:
        """Validate Office document."""
        try:
            # Basic file existence and extension check
            return os.path.exists(file_path) and os.path.getsize(file_path) > 0
        except:
            return False
    
    def _validate_image(self, file_path: str) -> bool:
        """Validate image file."""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()
            return True
        except:
            return False
    
    def _validate_text(self, file_path: str) -> bool:
        """Validate text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)  # Read first 1KB to check if it's readable
            return True
        except:
            try:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    f.read(1024)
                return True
            except:
                return False