# services/processor/src/pipeline/zip_dataset_processor.py
import os
import zipfile
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime

from src.pipeline.document_pipeline import DocumentProcessingPipeline, ProcessingResult
from src.models.document import Document

@dataclass
class DocumentPair:
    """Represents a pair of metadata and content documents."""
    content_doc: Optional[Document] = None
    metadata_doc: Optional[Document] = None
    zip_file: str = ""
    category: str = ""
    base_name: str = ""
    extraction_path: str = ""
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class ZipDatasetProcessor:
    """Processor for zip-based datasets with paired PDF files."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the zip dataset processor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize the document processing pipeline
        self.pipeline = DocumentProcessingPipeline(config.get("pipeline", {}))
        
        # Processing statistics
        self.stats = {
            "zip_files_processed": 0,
            "zip_files_failed": 0,
            "document_pairs_created": 0,
            "content_docs_processed": 0,
            "metadata_docs_processed": 0,
            "by_category": {},
            "errors": []
        }
        
        # Temporary extraction directory
        self.temp_dir = tempfile.mkdtemp(prefix="zip_dataset_")
        self.logger.info(f"Using temporary directory: {self.temp_dir}")
    
    def process_dataset(self, input_root: str, output_dir: Optional[str] = None) -> List[DocumentPair]:
        """Process the entire dataset structure."""
        
        self.logger.info(f"Starting dataset processing from: {input_root}")
        
        if not os.path.exists(input_root):
            raise ValueError(f"Input directory does not exist: {input_root}")
        
        # Discover all zip files organized by category
        zip_files_by_category = self._discover_zip_files(input_root)
        
        total_zips = sum(len(files) for files in zip_files_by_category.values())
        self.logger.info(f"Found {total_zips} zip files across {len(zip_files_by_category)} categories")
        
        # Process all zip files
        all_document_pairs = []
        
        for category, zip_files in zip_files_by_category.items():
            self.logger.info(f"Processing {len(zip_files)} zip files in category: {category}")
            
            category_pairs = self._process_category(category, zip_files)
            all_document_pairs.extend(category_pairs)
            
            # Update statistics
            self.stats["by_category"][category] = {
                "zip_files": len(zip_files),
                "document_pairs": len(category_pairs),
                "successful_pairs": len([p for p in category_pairs if p.content_doc and p.metadata_doc])
            }
        
        # Save processing results if output directory specified
        if output_dir:
            self._save_results(all_document_pairs, output_dir)
        
        self.logger.info(f"Dataset processing complete. Processed {len(all_document_pairs)} document pairs")
        
        return all_document_pairs
    
    def _discover_zip_files(self, input_root: str) -> Dict[str, List[str]]:
        """Discover all zip files organized by category."""
        
        zip_files_by_category = {}
        
        # Walk through the directory structure
        for category_dir in os.listdir(input_root):
            category_path = os.path.join(input_root, category_dir)
            
            if not os.path.isdir(category_path):
                continue
            
            # Skip hidden directories and files
            if category_dir.startswith('.'):
                continue
                
            zip_files = []
            
            # Find all zip files in this category
            for file_name in os.listdir(category_path):
                if file_name.endswith('.zip'):
                    zip_path = os.path.join(category_path, file_name)
                    zip_files.append(zip_path)
            
            if zip_files:
                zip_files_by_category[category_dir] = sorted(zip_files)
                self.logger.info(f"Category '{category_dir}': {len(zip_files)} zip files")
        
        return zip_files_by_category
    
    def _process_category(self, category: str, zip_files: List[str]) -> List[DocumentPair]:
        """Process all zip files in a category."""
        
        document_pairs = []
        
        # Process zip files in parallel
        max_workers = self.config.get("max_workers", 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all zip processing jobs
            future_to_zip = {
                executor.submit(self._process_zip_file, zip_file, category): zip_file
                for zip_file in zip_files
            }
            
            # Collect results
            for future in as_completed(future_to_zip):
                zip_file = future_to_zip[future]
                
                try:
                    document_pair = future.result()
                    document_pairs.append(document_pair)
                    
                    if document_pair.content_doc and document_pair.metadata_doc:
                        self.logger.info(f"Successfully processed zip: {os.path.basename(zip_file)}")
                        self.stats["zip_files_processed"] += 1
                    else:
                        self.logger.warning(f"Partial processing for zip: {os.path.basename(zip_file)}")
                        if document_pair.errors:
                            self.logger.warning(f"Errors: {'; '.join(document_pair.errors)}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to process zip {zip_file}: {e}")
                    self.stats["zip_files_failed"] += 1
                    self.stats["errors"].append(f"{zip_file}: {str(e)}")
                    
                    # Create error document pair
                    document_pairs.append(DocumentPair(
                        zip_file=zip_file,
                        category=category,
                        base_name=os.path.splitext(os.path.basename(zip_file))[0],
                        errors=[str(e)]
                    ))
        
        return document_pairs
    
    def _process_zip_file(self, zip_file: str, category: str) -> DocumentPair:
        """Process a single zip file containing PDF pair."""
        
        base_name = os.path.splitext(os.path.basename(zip_file))[0]
        
        # Create document pair
        document_pair = DocumentPair(
            zip_file=zip_file,
            category=category,
            base_name=base_name
        )
        
        try:
            # Extract zip file
            extraction_path = self._extract_zip_file(zip_file, base_name)
            document_pair.extraction_path = extraction_path
            
            # Find PDF files in extracted content
            pdf_files = self._find_pdf_files(extraction_path)
            
            if len(pdf_files) != 2:
                document_pair.errors.append(f"Expected 2 PDF files, found {len(pdf_files)}: {pdf_files}")
                return document_pair
            
            # Identify metadata and content PDFs
            metadata_pdf, content_pdf = self._identify_pdf_pair(pdf_files, base_name)
            
            if not metadata_pdf or not content_pdf:
                document_pair.errors.append("Could not identify metadata and content PDFs")
                return document_pair
            
            # Process both PDFs
            metadata_result = self._process_pdf_with_context(
                metadata_pdf, category, "metadata", base_name
            )
            content_result = self._process_pdf_with_context(
                content_pdf, category, "content", base_name
            )
            
            # Store results
            if metadata_result.success:
                document_pair.metadata_doc = metadata_result.document
                self.stats["metadata_docs_processed"] += 1
            else:
                document_pair.errors.append(f"Metadata PDF processing failed: {metadata_result.error}")
            
            if content_result.success:
                document_pair.content_doc = content_result.document
                self.stats["content_docs_processed"] += 1
            else:
                document_pair.errors.append(f"Content PDF processing failed: {content_result.error}")
            
            # Link the documents if both successful
            if document_pair.content_doc and document_pair.metadata_doc:
                self._link_document_pair(document_pair)
                self.stats["document_pairs_created"] += 1
            
        except Exception as e:
            document_pair.errors.append(str(e))
            self.logger.error(f"Error processing zip {zip_file}: {e}")
        
        finally:
            # Clean up extracted files
            if document_pair.extraction_path and os.path.exists(document_pair.extraction_path):
                try:
                    shutil.rmtree(document_pair.extraction_path)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {document_pair.extraction_path}: {e}")
        
        return document_pair
    
    def _extract_zip_file(self, zip_file: str, base_name: str) -> str:
        """Extract zip file to temporary directory."""
        
        extraction_path = os.path.join(self.temp_dir, f"{base_name}_{uuid.uuid4().hex[:8]}")
        os.makedirs(extraction_path, exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
        
        return extraction_path
    
    def _find_pdf_files(self, extraction_path: str) -> List[str]:
        """Find all PDF files in the extracted directory."""
        
        pdf_files = []
        
        for root, _, files in os.walk(extraction_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        return pdf_files
    
    def _identify_pdf_pair(self, pdf_files: List[str], base_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Identify which PDF is metadata and which is content."""
        
        metadata_pdf = None
        content_pdf = None
        
        for pdf_file in pdf_files:
            file_name = os.path.basename(pdf_file)
            name_without_ext = os.path.splitext(file_name)[0]
            
            # Check if this is the metadata file (contains -info suffix)
            if name_without_ext.endswith('-info') or 'info' in name_without_ext.lower():
                metadata_pdf = pdf_file
            else:
                # Check if this matches the base name or is the main content
                if name_without_ext == base_name or name_without_ext.startswith(base_name):
                    content_pdf = pdf_file
        
        # If we couldn't identify by naming convention, use file size heuristic
        if not metadata_pdf or not content_pdf:
            # Sort by file size - metadata is usually smaller
            pdf_files_with_size = [(f, os.path.getsize(f)) for f in pdf_files]
            pdf_files_with_size.sort(key=lambda x: x[1])
            
            # Smaller file is likely metadata
            metadata_pdf = pdf_files_with_size[0][0]
            content_pdf = pdf_files_with_size[1][0]
        
        return metadata_pdf, content_pdf
    
    def _process_pdf_with_context(self, pdf_file: str, category: str, doc_type: str, base_name: str) -> ProcessingResult:
        """Process PDF with additional context metadata."""
        
        # Add context metadata
        context_metadata = {
            "dataset_category": category,
            "document_type": doc_type,  # "metadata" or "content"
            "base_name": base_name,
            "source_zip": f"{base_name}.zip",
            "is_paired_document": True
        }
        
        # Process the PDF
        result = self.pipeline.process_file(pdf_file, context_metadata)
        
        return result
    
    def _link_document_pair(self, document_pair: DocumentPair):
        """Create bidirectional links between paired documents."""
        
        if not document_pair.content_doc or not document_pair.metadata_doc:
            return
        
        # Add relationship metadata
        document_pair.content_doc.metadata.update({
            "metadata_document_id": document_pair.metadata_doc.doc_id,
            "has_metadata_document": True
        })
        
        document_pair.metadata_doc.metadata.update({
            "content_document_id": document_pair.content_doc.doc_id,
            "describes_document": True
        })
        
        # Add to content types
        if "paired" not in document_pair.content_doc.content_type:
            document_pair.content_doc.content_type.append("paired")
        
        if "metadata" not in document_pair.metadata_doc.content_type:
            document_pair.metadata_doc.content_type.append("metadata")
    
    def _save_results(self, document_pairs: List[DocumentPair], output_dir: str):
        """Save processing results to output directory."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary statistics
        summary = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_document_pairs": len(document_pairs),
            "successful_pairs": len([p for p in document_pairs if p.content_doc and p.metadata_doc]),
            "statistics": self.stats,
            "document_pairs": []
        }
        
        # Save detailed results
        for pair in document_pairs:
            pair_data = {
                "base_name": pair.base_name,
                "category": pair.category,
                "zip_file": pair.zip_file,
                "errors": pair.errors,
                "content_document": {
                    "doc_id": pair.content_doc.doc_id if pair.content_doc else None,
                    "file_name": pair.content_doc.file_name if pair.content_doc else None,
                    "content_types": pair.content_doc.content_type if pair.content_doc else None,
                    "chunks_count": len(pair.content_doc.chunks) if pair.content_doc else 0
                },
                "metadata_document": {
                    "doc_id": pair.metadata_doc.doc_id if pair.metadata_doc else None,
                    "file_name": pair.metadata_doc.file_name if pair.metadata_doc else None,
                    "content_types": pair.metadata_doc.content_type if pair.metadata_doc else None,
                    "chunks_count": len(pair.metadata_doc.chunks) if pair.metadata_doc else 0
                }
            }
            summary["document_pairs"].append(pair_data)
        
        # Save to file
        summary_file = os.path.join(output_dir, f"dataset_processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Results saved to: {summary_file}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp directory: {e}")

# Integration with storage systems
class ZipDatasetStorage:
    """Storage integration for processed zip dataset."""
    
    def __init__(self, metadata_storage, vector_store, graph_storage):
        """Initialize with storage backends."""
        self.metadata_storage = metadata_storage
        self.vector_store = vector_store
        self.graph_storage = graph_storage
        self.logger = logging.getLogger(__name__)
    
    def store_document_pairs(self, document_pairs: List[DocumentPair]) -> Dict[str, Any]:
        """Store all document pairs in storage systems."""
        
        results = {
            "stored_content_docs": 0,
            "stored_metadata_docs": 0,
            "storage_errors": [],
            "relationships_created": 0
        }
        
        for pair in document_pairs:
            try:
                # Store content document
                if pair.content_doc:
                    self._store_single_document(pair.content_doc)
                    results["stored_content_docs"] += 1
                
                # Store metadata document
                if pair.metadata_doc:
                    self._store_single_document(pair.metadata_doc)
                    results["stored_metadata_docs"] += 1
                
                # Create relationships in graph database
                if pair.content_doc and pair.metadata_doc:
                    self._create_document_relationships(pair)
                    results["relationships_created"] += 1
                    
            except Exception as e:
                error_msg = f"Storage error for {pair.base_name}: {str(e)}"
                self.logger.error(error_msg)
                results["storage_errors"].append(error_msg)
        
        return results
    
    def _store_single_document(self, document: Document):
        """Store a single document in all storage systems."""
        
        # Store metadata in DuckDB
        self.metadata_storage.store_document(document)
        
        # Store chunks in vector store
        if document.chunks:
            self.vector_store.store_chunks(document.chunks)
        
        # Store document in graph database
        self.graph_storage.store_document_relations(document.__dict__)
    
    def _create_document_relationships(self, pair: DocumentPair):
        """Create special relationships for document pairs in graph database."""
        
        if not pair.content_doc or not pair.metadata_doc:
            return
        
        # Create metadata relationship
        relationship_data = {
            "source_doc_id": pair.metadata_doc.doc_id,
            "target_doc_id": pair.content_doc.doc_id,
            "relationship_type": "DESCRIBES",
            "relationship_metadata": {
                "category": pair.category,
                "base_name": pair.base_name,
                "created_at": datetime.now().isoformat()
            }
        }
        
        # Store in graph database (you may need to implement this method)
        # self.graph_storage.create_custom_relationship(relationship_data)


# Example usage script
def main():
    """Example usage of the ZIP dataset processor."""
    
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = {
        "max_workers": 4,
        "pipeline": {
            "pdf": {
                "engine": "pymupdf",
                "extract_images": True,
                "perform_ocr": True
            },
            "chunker": {
                "default_chunk_size": 500,
                "default_chunk_overlap": 50
            }
        }
    }
    
    # Initialize processor
    processor = ZipDatasetProcessor(config)
    
    try:
        # Process the dataset
        input_root = "/path/to/your/dataset"  # Update this path
        output_dir = "/path/to/output"        # Update this path
        
        document_pairs = processor.process_dataset(input_root, output_dir)
        
        # Print statistics
        stats = processor.get_statistics()
        print("\nProcessing Statistics:")
        print(f"Total zip files processed: {stats['zip_files_processed']}")
        print(f"Total zip files failed: {stats['zip_files_failed']}")
        print(f"Document pairs created: {stats['document_pairs_created']}")
        print(f"Content documents processed: {stats['content_docs_processed']}")
        print(f"Metadata documents processed: {stats['metadata_docs_processed']}")
        
        print("\nBy Category:")
        for category, cat_stats in stats['by_category'].items():
            print(f"  {category}: {cat_stats['successful_pairs']}/{cat_stats['zip_files']} successful")
        
        # Store in databases (optional)
        # from src.storage.duckdb import DuckDBStorage
        # from src.storage.vector_store import VectorStore
        # from src.storage.graph_db import GraphDBStorage
        # 
        # storage = ZipDatasetStorage(
        #     DuckDBStorage("/data/metadata.db"),
        #     VectorStore({"host": "chroma", "port": 8000}),
        #     GraphDBStorage({"uri": "bolt://neo4j:7687", "user": "neo4j", "password": "password"})
        # )
        # 
        # storage_results = storage.store_document_pairs(document_pairs)
        # print(f"\nStorage Results: {storage_results}")
        
    finally:
        # Clean up
        processor.cleanup()

if __name__ == "__main__":
    main()