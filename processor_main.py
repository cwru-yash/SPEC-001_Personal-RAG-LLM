#!/usr/bin/env python3
"""
Fixed main script for document processing pipeline.
"""
import os
import sys
import logging
from datetime import datetime

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main entry point with fixed imports."""
    logger = setup_logging()
    logger.info("Starting document processing pipeline")
    
    try:
        # Import with the corrected path
        from services.processor.src.pipeline.zip_dataset_processor import ZipDatasetProcessor
        
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
        
        # Set input/output paths
        input_path = os.path.join(current_dir, "data", "input")
        output_path = os.path.join(current_dir, "data", "processed")
        
        # Ensure directories exist
        os.makedirs(input_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        
        logger.info(f"Processing files from {input_path}")
        
        # Process the dataset
        document_pairs = processor.process_dataset(input_path, output_path)
        
        # Print statistics
        stats = processor.get_statistics()
        logger.info(f"Total zip files processed: {stats['zip_files_processed']}")
        logger.info(f"Document pairs created: {stats['document_pairs_created']}")
        
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main()