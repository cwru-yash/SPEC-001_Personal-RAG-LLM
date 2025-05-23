# services/processor/src/main.py
import hydra
from omegaconf import DictConfig
import os
import sys
from typing import List
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.document_pipeline import DocumentProcessingPipeline, ProcessingResult
from src.storage.duckdb import DuckDBStorage
from src.storage.vector_store import VectorStore
from src.storage.graph_db import GraphDBStorage

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for the document processing pipeline."""
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting document processing pipeline with config: {cfg.app_name} v{cfg.app_version}")
    
    # Initialize the processing pipeline
    pipeline = DocumentProcessingPipeline(cfg)
    
    # Initialize storage components
    try:
        # Initialize metadata storage (DuckDB)
        logger.info("Initializing DuckDB storage...")
        metadata_storage = DuckDBStorage(cfg.storage.duckdb.database)
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore(cfg.storage.vector_store)
        
        # Initialize graph storage
        logger.info("Initializing graph database...")
        graph_storage = GraphDBStorage(cfg.storage.graph_db)
        
    except Exception as e:
        logger.error(f"Failed to initialize storage components: {e}")
        return
    
    # Get input directory
    input_dir = cfg.processing.input_dir
    if not os.path.exists(input_dir):
        logger.warning(f"Input directory not found: {input_dir}")
        logger.info(f"Creating input directory: {input_dir}")
        os.makedirs(input_dir, exist_ok=True)
        logger.info("Place your documents in the input directory and run again.")
        return
    
    # Process documents
    logger.info(f"Processing documents from: {input_dir}")
    logger.info(f"Supported file types: {pipeline.processor_registry.supported_types()}")
    
    # Process all files in the input directory
    results = pipeline.process_directory(
        input_dir, 
        recursive=True, 
        max_workers=cfg.processing.max_workers
    )
    
    # Store processed documents
    successful_documents = []
    failed_documents = []
    
    for result in results:
        if result.success and result.document:
            try:
                # Store document metadata in DuckDB
                metadata_storage.store_document(result.document)
                
                # Store chunks in vector store
                if result.document.chunks:
                    vector_store.store_chunks(result.document.chunks)
                
                # Store document relationships in graph database
                graph_storage.store_document_relations(result.document.__dict__)
                
                successful_documents.append(result)
                logger.info(f"Successfully stored: {result.document.file_name}")
                
            except Exception as e:
                logger.error(f"Failed to store document {result.document.file_name}: {e}")
                failed_documents.append(result)
        else:
            failed_documents.append(result)
            logger.error(f"Failed to process document: {result.error}")
    
    # Print processing statistics
    stats = pipeline.get_statistics()
    logger.info("\n" + "="*50)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*50)
    logger.info(f"Total files processed: {stats['total_files']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Success rate: {stats['success_rate']:.2%}")
    logger.info(f"Average processing time: {stats['average_processing_time']:.2f}s")
    logger.info(f"Total processing time: {stats['total_processing_time']:.2f}s")
    
    if stats['by_file_type']:
        logger.info("\nBy file type:")
        for file_type, type_stats in stats['by_file_type'].items():
            total = type_stats['processed'] + type_stats['failed']
            success_rate = type_stats['processed'] / total if total > 0 else 0
            avg_time = type_stats['total_time'] / total if total > 0 else 0
            logger.info(f"  {file_type}: {type_stats['processed']}/{total} ({success_rate:.1%}) - avg: {avg_time:.2f}s")
    
    # Store processing results for analysis
    if cfg.get("save_results", True):
        save_processing_results(successful_documents, failed_documents, cfg)
    
    logger.info("Processing pipeline completed.")

def save_processing_results(successful: List[ProcessingResult], failed: List[ProcessingResult], cfg: DictConfig):
    """Save processing results for analysis."""
    import json
    from datetime import datetime
    
    output_dir = cfg.processing.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"processing_results_{timestamp}.json")
    
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "input_dir": cfg.processing.input_dir,
            "max_workers": cfg.processing.max_workers,
            "supported_types": cfg.processing.supported_file_types
        },
        "summary": {
            "total": len(successful) + len(failed),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / (len(successful) + len(failed)) if (len(successful) + len(failed)) > 0 else 0
        },
        "successful_documents": [
            {
                "file_name": result.document.file_name,
                "doc_id": result.document.doc_id,
                "content_types": result.document.content_type,
                "chunks_created": len(result.document.chunks),
                "processing_time": result.processing_time,
                "metadata": result.metadata
            }
            for result in successful if result.document
        ],
        "failed_documents": [
            {
                "error": result.error,
                "processing_time": result.processing_time
            }
            for result in failed
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Processing results saved to: {results_file}")

# Custom processor registration example
def register_custom_processors(pipeline: DocumentProcessingPipeline):
    """Example of how to register custom processors."""
    
    # Example: Custom processor for .log files
    def log_file_processor(file_path: str):
        """Custom processor for log files."""
        import uuid
        from datetime import datetime
        from src.models.document import Document
        
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Parse log entries (example for common log format)
        log_entries = []
        for line in content.splitlines():
            if line.strip() and not line.startswith('#'):
                log_entries.append(line)
        
        document = Document(
            doc_id=doc_id,
            file_name=file_name,
            file_extension="log",
            content_type=["log", "text"],
            created_at=datetime.now(),
            text_content=content,
            metadata={
                "log_entries": len(log_entries),
                "file_size": len(content)
            }
        )
        
        return document
    
    # Register the custom processor
    pipeline.register_custom_processor("log", log_file_processor)

if __name__ == "__main__":
    main()


# services/processor/src/pipeline/pipeline_extensions.py
"""Pipeline extensions for adding custom processing capabilities."""

import os
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

from src.models.document import Document
from src.pipeline.document_pipeline import DocumentProcessingPipeline

@dataclass
class ProcessingHook:
    """Hook for adding custom processing steps."""
    name: str
    function: Callable
    stage: str  # 'pre_process', 'post_extract', 'post_chunk', 'post_classify'
    priority: int = 0  # Higher priority runs first

class PipelineExtensions:
    """Extensions for the document processing pipeline."""
    
    def __init__(self):
        self.hooks: Dict[str, List[ProcessingHook]] = {
            'pre_process': [],
            'post_extract': [],
            'post_chunk': [],
            'post_classify': []
        }
    
    def register_hook(self, hook: ProcessingHook):
        """Register a processing hook."""
        if hook.stage not in self.hooks:
            raise ValueError(f"Invalid hook stage: {hook.stage}")
        
        self.hooks[hook.stage].append(hook)
        # Sort by priority (higher first)
        self.hooks[hook.stage].sort(key=lambda x: x.priority, reverse=True)
    
    def execute_hooks(self, stage: str, document: Document, **kwargs) -> Document:
        """Execute all hooks for a given stage."""
        for hook in self.hooks.get(stage, []):
            try:
                document = hook.function(document, **kwargs)
            except Exception as e:
                print(f"Error in hook {hook.name}: {e}")
        
        return document

# Example custom processors
class CustomProcessors:
    """Collection of example custom processors."""
    
    @staticmethod
    def code_file_processor(file_path: str) -> Document:
        """Processor for source code files."""
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1][1:].lower()
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Analyze code structure
        lines = content.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        comment_lines = [line for line in lines if line.strip().startswith(('#', '//', '/*', '*', '--'))]
        
        document = Document(
            doc_id=doc_id,
            file_name=file_name,
            file_extension=file_extension,
            content_type=["code", file_extension],
            created_at=datetime.now(),
            text_content=content,
            metadata={
                "total_lines": len(lines),
                "code_lines": len(non_empty_lines),
                "comment_lines": len(comment_lines),
                "programming_language": file_extension,
                "file_size": len(content)
            }
        )
        
        return document
    
    @staticmethod
    def config_file_processor(file_path: str) -> Document:
        """Processor for configuration files."""
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1][1:].lower()
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Basic config analysis
        lines = content.splitlines()
        config_entries = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                config_entries.append({
                    "key": key.strip(),
                    "value": value.strip()
                })
        
        document = Document(
            doc_id=doc_id,
            file_name=file_name,
            file_extension=file_extension,
            content_type=["config", file_extension],
            created_at=datetime.now(),
            text_content=content,
            metadata={
                "config_type": file_extension,
                "config_entries": len(config_entries),
                "total_lines": len(lines)
            }
        )
        
        return document

# Example hooks
def document_validation_hook(document: Document, **kwargs) -> Document:
    """Hook to validate document after extraction."""
    if not document.text_content.strip():
        document.metadata["validation_warning"] = "No text content extracted"
    
    if len(document.text_content) > 1_000_000:  # 1MB of text
        document.metadata["validation_warning"] = "Very large document"
    
    return document

def content_enhancement_hook(document: Document, **kwargs) -> Document:
    """Hook to enhance document content after extraction."""
    # Add word count
    if document.text_content:
        word_count = len(document.text_content.split())
        document.metadata["word_count"] = word_count
        
        # Classify document size
        if word_count < 100:
            document.metadata["size_category"] = "short"
        elif word_count < 1000:
            document.metadata["size_category"] = "medium"
        else:
            document.metadata["size_category"] = "long"
    
    return document

def quality_assessment_hook(document: Document, **kwargs) -> Document:
    """Hook to assess document quality."""
    quality_score = 1.0
    issues = []
    
    # Check for extraction errors
    if "extraction_error" in document.metadata:
        quality_score -= 0.5
        issues.append("extraction_error")
    
    # Check text content
    if not document.text_content.strip():
        quality_score -= 0.3
        issues.append("no_text_content")
    
    # Check for very short content
    if len(document.text_content) < 50:
        quality_score -= 0.2
        issues.append("very_short_content")
    
    document.metadata["quality_score"] = max(0.0, quality_score)
    if issues:
        document.metadata["quality_issues"] = issues
    
    return document

# Usage example
def setup_extended_pipeline(config: Dict[str, Any]) -> DocumentProcessingPipeline:
    """Setup pipeline with custom processors and hooks."""
    
    # Create base pipeline
    pipeline = DocumentProcessingPipeline(config)
    
    # Register custom processors
    code_extensions = ['py', 'js', 'ts', 'java', 'cpp', 'c', 'h', 'go', 'rs', 'php']
    for ext in code_extensions:
        pipeline.register_custom_processor(ext, CustomProcessors.code_file_processor)
    
    config_extensions = ['conf', 'ini', 'env', 'properties', 'cfg']
    for ext in config_extensions:
        pipeline.register_custom_processor(ext, CustomProcessors.config_file_processor)
    
    # Setup extensions
    extensions = PipelineExtensions()
    
    # Register hooks
    extensions.register_hook(ProcessingHook(
        name="document_validation",
        function=document_validation_hook,
        stage="post_extract",
        priority=100
    ))
    
    extensions.register_hook(ProcessingHook(
        name="content_enhancement",
        function=content_enhancement_hook,
        stage="post_extract",
        priority=50
    ))
    
    extensions.register_hook(ProcessingHook(
        name="quality_assessment",
        function=quality_assessment_hook,
        stage="post_classify",
        priority=10
    ))
    
    # Monkey patch the pipeline to use extensions
    original_process_file = pipeline.process_file
    
    def extended_process_file(file_path: str, metadata: Optional[Dict[str, Any]] = None):
        # Pre-process hooks
        dummy_doc = Document(
            doc_id="temp",
            file_name=os.path.basename(file_path),
            file_extension="",
            content_type=[],
            text_content="",
            metadata=metadata or {}
        )
        dummy_doc = extensions.execute_hooks("pre_process", dummy_doc, file_path=file_path)
        
        # Original processing
        result = original_process_file(file_path, dummy_doc.metadata)
        
        if result.success and result.document:
            # Post-process hooks
            result.document = extensions.execute_hooks("post_extract", result.document)
            result.document = extensions.execute_hooks("post_chunk", result.document)
            result.document = extensions.execute_hooks("post_classify", result.document)
        
        return result
    
    pipeline.process_file = extended_process_file
    
    return pipeline