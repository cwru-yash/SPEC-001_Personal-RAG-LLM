# services/processor/src/pipeline/enhanced_zip_processor.py
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

from .document_pipeline import DocumentProcessingPipeline, ProcessingResult
from ..models.document import Document

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

class EnhancedZipProcessor:
    """Enhanced ZIP processor that handles all file types including presentations and images."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced zip processor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize the document processing pipeline with all extractors
        self.pipeline = DocumentProcessingPipeline(config.get("pipeline", {}))
        
        # Log supported file types
        supported_types = self.pipeline.processor_registry.supported_types()
        self.logger.info(f"Pipeline supports: {supported_types}")
        
        # Processing statistics
        self.stats = {
            "zip_files_processed": 0,
            "zip_files_failed": 0,
            "document_pairs_created": 0,
            "content_docs_processed": 0,
            "metadata_docs_processed": 0,
            "by_category": {},
            "by_file_type": {},
            "errors": []
        }
        
        # Temporary extraction directory
        self.temp_dir = tempfile.mkdtemp(prefix="enhanced_zip_")
        self.logger.info(f"Using temporary directory: {self.temp_dir}")
    
    def process_dataset(self, input_root: str, output_dir: Optional[str] = None) -> List[DocumentPair]:
        """Process the entire dataset structure."""
        
        self.logger.info(f"Starting enhanced dataset processing from: {input_root}")
        
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
            successful_pairs = len([p for p in category_pairs if p.content_doc and p.metadata_doc])
            self.stats["by_category"][category] = {
                "zip_files": len(zip_files),
                "document_pairs": len(category_pairs),
                "successful_pairs": successful_pairs
            }
        
        # Save processing results if output directory specified
        if output_dir:
            self._save_results(all_document_pairs, output_dir)
        
        self.logger.info(f"Enhanced dataset processing complete. Processed {len(all_document_pairs)} document pairs")
        
        return all_document_pairs
    
    def _discover_zip_files(self, input_root: str) -> Dict[str, List[str]]:
        """Discover all zip files organized by category."""
        
        zip_files_by_category = {}
        
        for category_dir in os.listdir(input_root):
            category_path = os.path.join(input_root, category_dir)
            
            if not os.path.isdir(category_path) or category_dir.startswith('.'):
                continue
                
            zip_files = []
            
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
        max_workers = self.config.get("max_workers", 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_zip = {
                executor.submit(self._process_zip_file, zip_file, category): zip_file
                for zip_file in zip_files
            }
            
            for future in as_completed(future_to_zip):
                zip_file = future_to_zip[future]
                
                try:
                    document_pair = future.result()
                    document_pairs.append(document_pair)
                    
                    if document_pair.content_doc and document_pair.metadata_doc:
                        self.logger.info(f"✅ Successfully processed: {os.path.basename(zip_file)}")
                        self.stats["zip_files_processed"] += 1
                    else:
                        self.logger.warning(f"⚠️ Partial processing: {os.path.basename(zip_file)}")
                        if document_pair.errors:
                            self.logger.warning(f"   Errors: {'; '.join(document_pair.errors)}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Failed to process zip {zip_file}: {e}")
                    self.stats["zip_files_failed"] += 1
                    self.stats["errors"].append(f"{zip_file}: {str(e)}")
                    
                    document_pairs.append(DocumentPair(
                        zip_file=zip_file,
                        category=category,
                        base_name=os.path.splitext(os.path.basename(zip_file))[0],
                        errors=[str(e)]
                    ))
        
        return document_pairs
    
    def _process_zip_file(self, zip_file: str, category: str) -> DocumentPair:
        """Process a single zip file with enhanced support for all file types."""
        
        base_name = os.path.splitext(os.path.basename(zip_file))[0]
        
        document_pair = DocumentPair(
            zip_file=zip_file,
            category=category,
            base_name=base_name
        )
        
        try:
            # Extract zip file with nested handling
            extraction_path = self._extract_zip_with_nesting(zip_file, base_name)
            document_pair.extraction_path = extraction_path
            
            # Find PDF files in extracted content
            pdf_files = self._find_pdf_files(extraction_path)

            if len(pdf_files) != 2:
                document_pair.errors.append(f"Expected 2 PDF files, found {len(pdf_files)}: {pdf_files}")
                return document_pair

            # With this:
            # Find all processable files in extracted content
            processable_files = self._find_content_files(extraction_path)

            if len(processable_files) == 0:
                document_pair.errors.append(f"No processable files found in ZIP")
                return document_pair

            # # Find all processable files (not just PDFs)
            # processable_files = self._find_processable_files(extraction_path)
            
            if len(processable_files) == 0:
                document_pair.errors.append("No processable files found in ZIP")
                return document_pair
            
            # Log what we found
            file_types = [Path(f).suffix.lower() for f in processable_files]
            self.logger.debug(f"Found files in {base_name}: {file_types}")
            
            # Identify metadata and content files based on category
            # metadata_file, content_file = self._identify_file_pair(processable_files, base_name, category)
            
            # Identify metadata and content files based on category
            if category == "Presentations":
                metadata_file, content_file = self._identify_presentation_pair(processable_files, base_name)
            elif category == "Images":
                metadata_file, content_file = self._identify_image_pair(processable_files, base_name)
            elif category == "Spreadsheets":
                metadata_file, content_file = self._identify_spreadsheet_pair(processable_files, base_name)
            else:
                # Default to PDF pair identification
                pdf_files = [f for f in processable_files if f.lower().endswith('.pdf')]
                if len(pdf_files) >= 2:
                    metadata_file, content_file = self._identify_pdf_pair(pdf_files, base_name)
                else:
                    # Fallback for mixed file types
                    metadata_file, content_file = self._identify_mixed_pair(processable_files, base_name)

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
                    
                    # Track file types
                    file_ext = Path(content_file).suffix.lower()
                    if file_ext not in self.stats["by_file_type"]:
                        self.stats["by_file_type"][file_ext] = 0
                    self.stats["by_file_type"][file_ext] += 1
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
            # Clean up extracted files
            if document_pair.extraction_path and os.path.exists(document_pair.extraction_path):
                try:
                    shutil.rmtree(document_pair.extraction_path)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {document_pair.extraction_path}: {e}")
        
        return document_pair
    
    def _extract_zip_with_nesting(self, zip_file: str, base_name: str) -> str:
        """Extract zip file with support for nested archives."""
        
        extraction_path = os.path.join(self.temp_dir, f"{base_name}_{uuid.uuid4().hex[:8]}")
        os.makedirs(extraction_path, exist_ok=True)
        
        # Extract main archive
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
        
        # Handle nested ZIP files (common in Spreadsheets category)
        for root, _, files in os.walk(extraction_path):
            for file in files:
                if file.endswith('.zip'):
                    nested_zip = os.path.join(root, file)
                    nested_extract_dir = os.path.join(root, f"{os.path.splitext(file)[0]}_extracted")
                    
                    try:
                        with zipfile.ZipFile(nested_zip, 'r') as nested_zip_ref:
                            nested_zip_ref.extractall(nested_extract_dir)
                        os.remove(nested_zip)
                        self.logger.debug(f"Extracted nested ZIP: {file}")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract nested ZIP {file}: {e}")
        
        return extraction_path
    
    def _find_processable_files(self, extraction_path: str) -> List[str]:
        """Find all files that can be processed by the pipeline."""
        
        processable_files = []
        supported_extensions = set(self.pipeline.processor_registry.supported_types())
        
        for root, _, files in os.walk(extraction_path):
            for file in files:
                file_ext = Path(file).suffix[1:].lower()  # Remove dot
                if file_ext in supported_extensions:
                    processable_files.append(os.path.join(root, file))
        
        return processable_files
    
    def _identify_presentation_pair(self, files: List[str], base_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Identify content and metadata files for presentations."""
        
        # Look for presentation files (content) and PDF info files (metadata)
        presentation_files = [f for f in files if f.lower().endswith(('.pptx', '.ppt'))]
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        
        content_file = None
        metadata_file = None
        
        if presentation_files:
            content_file = presentation_files[0]  # Use first presentation file as content
        
        # Find metadata file (typically PDF with 'info' in name)
        for pdf_file in pdf_files:
            if 'info' in os.path.basename(pdf_file).lower():
                metadata_file = pdf_file
                break
        
        # If no explicit metadata file found, use file size heuristic
        if not metadata_file and pdf_files:
            # Sort PDFs by size - metadata typically smaller
            pdf_files_with_size = [(f, os.path.getsize(f)) for f in pdf_files]
            pdf_files_with_size.sort(key=lambda x: x[1])
            metadata_file = pdf_files_with_size[0][0]
        
        return metadata_file, content_file

    def _identify_image_pair(self, files: List[str], base_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Identify content and metadata files for images."""
        
        # Look for image files (content) and PDF info files (metadata)
        image_files = [f for f in files if any(f.lower().endswith(ext) for ext in 
                        ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'])]
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        
        content_file = None
        metadata_file = None
        
        if image_files:
            content_file = image_files[0]  # Use first image file as content
        
        # Find metadata file (typically PDF with 'info' in name)
        for pdf_file in pdf_files:
            if 'info' in os.path.basename(pdf_file).lower():
                metadata_file = pdf_file
                break
        
        # If no explicit metadata file found, use file size heuristic
        if not metadata_file and pdf_files:
            metadata_file = pdf_files[0]
        
        return metadata_file, content_file

    def _identify_mixed_pair(self, files: List[str], base_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Identify content and metadata for mixed file types."""
        if len(files) == 1:
            # Only one file - treat as content
            return None, files[0]
        
        # Sort by size - metadata usually smaller
        files_by_size = sorted(files, key=lambda f: os.path.getsize(f))
        
        # Check for 'info' pattern in any file
        metadata_file = None
        for f in files:
            if 'info' in os.path.basename(f).lower():
                metadata_file = f
                break
        
        if metadata_file:
            # Find content file (different from metadata file)
            content_file = next((f for f in files if f != metadata_file), None)
        else:
            # Use size heuristic
            metadata_file = files_by_size[0]  # Smallest file
            content_file = files_by_size[-1]  # Largest file
        
        return metadata_file, content_file

    def _identify_file_pair(self, files: List[str], base_name: str, category: str) -> Tuple[Optional[str], Optional[str]]:
        """Identify content and metadata files based on category and naming."""
        
        if len(files) == 0:
            return None, None
        
        if len(files) == 1:
            # Single file - treat as content
            return None, files[0]
        
        metadata_file = None
        content_file = None
        
        # Separate files by type
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        image_files = [f for f in files if any(f.lower().endswith(ext) 
                      for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'])]
        presentation_files = [f for f in files if any(f.lower().endswith(ext) 
                             for ext in ['.pptx', '.ppt'])]
        spreadsheet_files = [f for f in files if any(f.lower().endswith(ext) 
                            for ext in ['.xlsx', '.xls', '.csv'])]
        
        # Category-specific logic
        if category == "Images":
            # Content: image file, Metadata: PDF with info
            if image_files:
                content_file = image_files[0]
            
            for pdf in pdf_files:
                if 'info' in os.path.basename(pdf).lower():
                    metadata_file = pdf
                    break
                    
        elif category == "Presentations":
            # Content: presentation file, Metadata: PDF with info
            if presentation_files:
                content_file = presentation_files[0]
            
            for pdf in pdf_files:
                if 'info' in os.path.basename(pdf).lower():
                    metadata_file = pdf
                    break
                    
        elif category == "Spreadsheets":
            # Content: spreadsheet file, Metadata: PDF with info
            if spreadsheet_files:
                content_file = spreadsheet_files[0]
            
            for pdf in pdf_files:
                if 'info' in os.path.basename(pdf).lower():
                    metadata_file = pdf
                    break
                    
        else:
            # Default logic for Documents/Emails - use PDF pair identification
            if len(pdf_files) >= 2:
                metadata_file, content_file = self._identify_pdf_pair(pdf_files, base_name)
        
        # Fallback: use size heuristic if no specific identification
        if not metadata_file and not content_file and len(files) >= 2:
            files_by_size = sorted(files, key=lambda f: os.path.getsize(f))
            metadata_file = files_by_size[0]  # Smaller file
            content_file = files_by_size[-1]   # Larger file
        
        return metadata_file, content_file
    
    def _identify_pdf_pair(self, pdf_files: List[str], base_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Identify PDF pair using original logic."""
        
        metadata_pdf = None
        content_pdf = None
        
        for pdf_file in pdf_files:
            file_name = os.path.basename(pdf_file)
            name_without_ext = os.path.splitext(file_name)[0]
            
            if name_without_ext.endswith('-info') or 'info' in name_without_ext.lower():
                metadata_pdf = pdf_file
            else:
                if name_without_ext == base_name or name_without_ext.startswith(base_name):
                    content_pdf = pdf_file
        
        # Fallback: use size heuristic
        if not metadata_pdf or not content_pdf:
            pdf_files_with_size = [(f, os.path.getsize(f)) for f in pdf_files]
            pdf_files_with_size.sort(key=lambda x: x[1])
            
            metadata_pdf = pdf_files_with_size[0][0]
            content_pdf = pdf_files_with_size[1][0]
        
        return metadata_pdf, content_pdf
    
    def _process_file_with_context(self, file_path: str, category: str, 
                                   doc_type: str, base_name: str) -> ProcessingResult:
        """Process any supported file type with context."""
        
        context_metadata = {
            "dataset_category": category,
            "document_type": doc_type,
            "base_name": base_name,
            "source_zip": f"{base_name}.zip",
            "is_paired_document": True
        }
        
        return self.pipeline.process_file(file_path, context_metadata)
    
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
        
        # Save summary with enhanced statistics
        summary = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_document_pairs": len(document_pairs),
            "successful_pairs": len([p for p in document_pairs if p.content_doc and p.metadata_doc]),
            "statistics": self.stats,
            "file_type_distribution": self.stats["by_file_type"],
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
                    "chunks_count": len(pair.content_doc.chunks) if pair.content_doc else 0,
                    "file_extension": pair.content_doc.file_extension if pair.content_doc else None
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
        summary_file = os.path.join(output_dir, f"enhanced_processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Enhanced results saved to: {summary_file}")
    
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