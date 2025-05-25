# process_complete_dataset.py - Complete integrated processor
import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

# Import all components
from services.processor.src.pipeline.zip_dataset_processor import (
    ZipDatasetProcessor, ZipDatasetStorage, DocumentPair
)
from services.processor.src.pipeline.document_pipeline import DocumentProcessingPipeline
from services.processor.src.storage.duckdb import DuckDBStorage
from services.processor.src.storage.vector_store import VectorStore
from services.processor.src.storage.graph_db import GraphDBStorage

# Import the new extractors
from services.processor.src.pipeline.extractors.office_extractor import OfficeExtractor
from services.processor.src.pipeline.extractors.text_extractor import TextExtractor

class EnhancedZipDatasetProcessor(ZipDatasetProcessor):
    """Enhanced processor that handles non-PDF files in ZIP archives."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced processor with additional extractors."""
        super().__init__(config)
        
        # Register Office extractors
        office_extractor = OfficeExtractor(config.get("office", {}))
        for ext in ["xlsx", "xls", "docx", "doc", "pptx", "ppt"]:
            self.pipeline.register_custom_processor(ext, office_extractor.extract)
        
        # Register text/CSV extractors
        text_extractor = TextExtractor(config.get("text", {}))
        for ext in ["csv", "txt", "json", "xml", "html"]:
            self.pipeline.register_custom_processor(ext, text_extractor.extract)
        
        self.logger.info("Registered additional extractors for Office and text files")
    
    def _find_processable_files(self, extraction_path: str) -> List[str]:
        """Find all processable files (not just PDFs)."""
        processable_files = []
        supported_extensions = self.pipeline.processor_registry.supported_types()
        
        for root, _, files in os.walk(extraction_path):
            for file in files:
                file_ext = Path(file).suffix[1:].lower()
                if file_ext in supported_extensions:
                    processable_files.append(os.path.join(root, file))
        
        return processable_files
    
    def _process_zip_file(self, zip_file: str, category: str) -> DocumentPair:
        """Enhanced processing for ZIP files with mixed content."""
        base_name = os.path.splitext(os.path.basename(zip_file))[0]
        
        document_pair = DocumentPair(
            zip_file=zip_file,
            category=category,
            base_name=base_name
        )
        
        try:
            # Extract zip file
            extraction_path = self._extract_zip_file(zip_file, base_name)
            document_pair.extraction_path = extraction_path
            
            # Handle nested ZIPs (common in Spreadsheets category)
            self._handle_nested_zips(extraction_path)
            
            # Find all processable files
            processable_files = self._find_processable_files(extraction_path)
            
            if len(processable_files) == 0:
                document_pair.errors.append(f"No processable files found in ZIP")
                return document_pair
            
            # Identify metadata and content files
            if category == "Spreadsheets":
                # Special handling for spreadsheets with mixed file types
                metadata_file, content_file = self._identify_spreadsheet_pair(
                    processable_files, base_name
                )
            else:
                # Standard PDF pair identification
                pdf_files = [f for f in processable_files if f.lower().endswith('.pdf')]
                if len(pdf_files) >= 2:
                    metadata_file, content_file = self._identify_pdf_pair(pdf_files, base_name)
                else:
                    # Fallback for single file or mixed types
                    metadata_file, content_file = self._identify_mixed_pair(
                        processable_files, base_name
                    )
            
            # Process both files
            if metadata_file:
                metadata_result = self._process_file_with_context(
                    metadata_file, category, "metadata", base_name
                )
                if metadata_result.success:
                    document_pair.metadata_doc = metadata_result.document
                    self.stats["metadata_docs_processed"] += 1
                else:
                    document_pair.errors.append(f"Metadata processing failed: {metadata_result.error}")
            
            if content_file:
                content_result = self._process_file_with_context(
                    content_file, category, "content", base_name
                )
                if content_result.success:
                    document_pair.content_doc = content_result.document
                    self.stats["content_docs_processed"] += 1
                else:
                    document_pair.errors.append(f"Content processing failed: {content_result.error}")
            
            # Link documents if both successful
            if document_pair.content_doc and document_pair.metadata_doc:
                self._link_document_pair(document_pair)
                self.stats["document_pairs_created"] += 1
            
        except Exception as e:
            document_pair.errors.append(str(e))
            self.logger.error(f"Error processing zip {zip_file}: {e}")
        
        finally:
            # Cleanup
            if document_pair.extraction_path and os.path.exists(document_pair.extraction_path):
                try:
                    import shutil
                    shutil.rmtree(document_pair.extraction_path)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up: {e}")
        
        return document_pair
    
    def _handle_nested_zips(self, directory: str):
        """Extract any nested ZIP files."""
        import zipfile
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.zip'):
                    nested_zip = os.path.join(root, file)
                    extract_dir = os.path.join(root, f"{os.path.splitext(file)[0]}_extracted")
                    
                    try:
                        with zipfile.ZipFile(nested_zip, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                        os.remove(nested_zip)
                        self.logger.info(f"Extracted nested ZIP: {file}")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract nested ZIP {file}: {e}")
    
    def _identify_spreadsheet_pair(self, files: List[str], base_name: str) -> tuple:
        """Identify content and metadata for spreadsheet files."""
        # Separate by file type
        excel_files = [f for f in files if any(f.lower().endswith(ext) for ext in ['.xlsx', '.xls'])]
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        
        content_file = None
        metadata_file = None
        
        # Content is usually the Excel/CSV file
        if excel_files:
            content_file = excel_files[0]
        elif csv_files:
            content_file = csv_files[0]
        
        # Metadata is usually the PDF info file
        for pdf in pdf_files:
            if 'info' in os.path.basename(pdf).lower():
                metadata_file = pdf
                break
        
        # If no metadata PDF, use the smaller file
        if not metadata_file and len(files) >= 2:
            files_by_size = sorted(files, key=lambda f: os.path.getsize(f))
            metadata_file = files_by_size[0]
            if not content_file:
                content_file = files_by_size[-1]
        
        return metadata_file, content_file
    
    def _identify_mixed_pair(self, files: List[str], base_name: str) -> tuple:
        """Identify pairs from mixed file types."""
        if len(files) == 1:
            # Single file - treat as content
            return None, files[0]
        
        # Sort by size - metadata usually smaller
        files_by_size = sorted(files, key=lambda f: os.path.getsize(f))
        
        # Check for 'info' pattern
        metadata_file = None
        for f in files:
            if 'info' in os.path.basename(f).lower():
                metadata_file = f
                break
        
        if metadata_file:
            content_file = next((f for f in files if f != metadata_file), None)
        else:
            # Use size heuristic
            metadata_file = files_by_size[0]
            content_file = files_by_size[-1]
        
        return metadata_file, content_file
    
    def _process_file_with_context(self, file_path: str, category: str, 
                                   doc_type: str, base_name: str):
        """Process any supported file type with context."""
        # Use the enhanced pipeline that supports multiple file types
        return self.pipeline.process_file(file_path, {
            "dataset_category": category,
            "document_type": doc_type,
            "base_name": base_name,
            "source_zip": f"{base_name}.zip",
            "is_paired_document": True
        })


def setup_logging(log_level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"dataset_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )


def create_enhanced_config():
    """Create configuration for enhanced dataset processing."""
    config = {
        # Parallel processing
        "max_workers": 4,
        
        # Pipeline configuration
        "pipeline": {
            # PDF processing
            "pdf": {
                "engine": "pymupdf",
                "extract_images": True,
                "perform_ocr": True,
                "enable_document_extractor": True,
                "enable_email_extractor": True,
                "enable_presentation_extractor": True
            },
            
            # OCR settings
            "ocr": {
                "engine": "tesseract",
                "languages": "eng",
                "tesseract_config": "--psm 3"
            },
            
            # Office document settings
            "office": {
                "excel": {
                    "max_rows_per_sheet": 1000,
                    "include_formulas": False
                },
                "word": {
                    "extract_tables": True
                },
                "powerpoint": {
                    "extract_slide_notes": True
                }
            },
            
            # Text/CSV settings
            "text": {
                "encoding_fallbacks": ["utf-8", "latin-1", "cp1252"],
                "csv": {
                    "max_preview_rows": 100,
                    "auto_detect_delimiter": True
                }
            },
            
            # Chunking strategy
            "chunker": {
                "default_chunk_size": 500,
                "default_chunk_overlap": 50,
                "email_chunk_size": 300,
                "spreadsheet_chunk_size": 400
            },
            
            # Classification
            "classifier": {
                "enabled": True
            }
        },
        
        # Storage configuration
        "storage": {
            "duckdb": {
                "database": "/data/metadata.db"
            },
            "vector_store": {
                "host": "localhost",  # Change to "chroma" for Docker
                "port": 8000
            },
            "graph_db": {
                "uri": "bolt://localhost:7687",  # Change to "neo4j:7687" for Docker
                "user": "neo4j",
                "password": "password",
                "database": "neo4j"  # Use default database
            }
        }
    }
    
    return config


def process_complete_dataset(input_path: str, output_path: str = None):
    """Process the complete dataset with all enhancements."""
    
    print("=" * 60)
    print("Enhanced Personal RAG-LLM Dataset Processor")
    print("=" * 60)
    
    # Setup
    setup_logging()
    config = create_enhanced_config()
    
    # Initialize enhanced processor
    print("🔧 Initializing enhanced dataset processor...")
    processor = EnhancedZipDatasetProcessor(config)
    
    try:
        # Process the dataset
        print(f"📁 Processing dataset from: {input_path}")
        
        if output_path:
            document_pairs = processor.process_dataset(input_path, output_path)
        else:
            document_pairs = processor.process_dataset(input_path)
        
        # Print statistics
        stats = processor.get_statistics()
        print("\n" + "=" * 40)
        print("📊 PROCESSING RESULTS")
        print("=" * 40)
        print(f"✅ Zip files processed: {stats['zip_files_processed']}")
        print(f"❌ Zip files failed: {stats['zip_files_failed']}")
        print(f"📄 Document pairs created: {stats['document_pairs_created']}")
        print(f"📝 Content documents: {stats['content_docs_processed']}")
        print(f"ℹ️  Metadata documents: {stats['metadata_docs_processed']}")
        
        print(f"\n📂 Results by Category:")
        for category, cat_stats in stats['by_category'].items():
            success_rate = (cat_stats['successful_pairs'] / cat_stats['zip_files'] 
                          if cat_stats['zip_files'] > 0 else 0)
            print(f"  {category:15}: {cat_stats['successful_pairs']:3}/{cat_stats['zip_files']:3} "
                  f"({success_rate:.1%})")
        
        # Store in databases
        print("\n💾 Storing documents in databases...")
        try:
            # Initialize storage systems
            metadata_storage = DuckDBStorage(config["storage"]["duckdb"]["database"])
            vector_store = VectorStore(config["storage"]["vector_store"])
            graph_storage = GraphDBStorage(config["storage"]["graph_db"])
            
            # Store documents
            storage = ZipDatasetStorage(metadata_storage, vector_store, graph_storage)
            storage_results = storage.store_document_pairs(document_pairs)
            
            print(f"✅ Content documents stored: {storage_results['stored_content_docs']}")
            print(f"✅ Metadata documents stored: {storage_results['stored_metadata_docs']}")
            print(f"🔗 Relationships created: {storage_results['relationships_created']}")
            
            if storage_results['storage_errors']:
                print(f"❌ Storage errors: {len(storage_results['storage_errors'])}")
                for error in storage_results['storage_errors'][:5]:
                    print(f"    {error}")
            
        except Exception as e:
            print(f"❌ Database storage failed: {e}")
            print("Documents were processed but not stored in databases.")
            import traceback
            traceback.print_exc()
        
        print(f"\n🎉 Processing complete!")
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        processor.cleanup()


def test_single_category(input_path: str, category: str):
    """Test processing for a single category."""
    
    category_path = os.path.join(input_path, category)
    if not os.path.exists(category_path):
        print(f"❌ Category path not found: {category_path}")
        return
    
    print(f"Testing category: {category}")
    
    # Create a temporary input with just this category
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        test_category_dir = os.path.join(temp_dir, category)
        os.makedirs(test_category_dir)
        
        # Copy a few test files
        import shutil
        test_files = []
        for file in os.listdir(category_path)[:3]:  # Test with first 3 files
            if file.endswith('.zip'):
                src = os.path.join(category_path, file)
                dst = os.path.join(test_category_dir, file)
                shutil.copy2(src, dst)
                test_files.append(file)
        
        print(f"Testing with files: {test_files}")
        
        # Process
        output_dir = os.path.join(temp_dir, "test_output")
        process_complete_dataset(temp_dir, output_dir)


def main():
    """Main entry point."""
    
    # Configuration
    INPUT_PATH = "/data/input"  # Update this to your dataset path
    OUTPUT_PATH = "/data/processed_results"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        INPUT_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_PATH = sys.argv[2]
    
    print(f"Input path: {INPUT_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    
    # Check if we should test a specific category
    if len(sys.argv) > 3 and sys.argv[3] == "--test-category":
        category = sys.argv[4] if len(sys.argv) > 4 else "Spreadsheets"
        test_single_category(INPUT_PATH, category)
    else:
        # Process complete dataset
        process_complete_dataset(INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()