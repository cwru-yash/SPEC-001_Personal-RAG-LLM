# process_dataset.py - Main runner script for your ZIP dataset
import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from services.processor.src.pipeline.zip_dataset_processor import ZipDatasetProcessor, ZipDatasetStorage
from services.processor.src.storage.duckdb import DuckDBStorage
from services.processor.src.storage.vector_store import VectorStore
from services.processor.src.storage.graph_db import GraphDBStorage

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

def create_config():
    """Create configuration for dataset processing."""
    config = {
        # Parallel processing
        "max_workers": 4,  # Adjust based on your system
        
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
            
            # OCR settings for image-heavy PDFs
            "ocr": {
                "engine": "tesseract",
                "languages": "eng",
                "tesseract_config": "--psm 3",
                "preprocess_image": True
            },
            
            # Chunking strategy
            "chunker": {
                "default_chunk_size": 500,
                "default_chunk_overlap": 50,
                "adaptive_chunking": True
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
                "host": "chroma",
                "port": 8000
            },
            "graph_db": {
                "uri": "bolt://neo4j:7687",
                "user": "neo4j",
                "password": "password",
                "database": "documents"
            }
        }
    }
    
    return config

def validate_dataset_structure(input_path):
    """Validate the dataset structure."""
    if not os.path.exists(input_path):
        raise ValueError(f"Dataset path does not exist: {input_path}")
    
    # Expected categories based on your image
    expected_categories = ["Documents", "Emails", "Images", "Presentations", "Spreadsheets"]
    found_categories = []
    
    for item in os.listdir(input_path):
        item_path = os.path.join(input_path, item)
        if os.path.isdir(item_path):
            found_categories.append(item)
    
    print(f"Found categories: {found_categories}")
    
    # Check for zip files in each category
    total_zips = 0
    for category in found_categories:
        category_path = os.path.join(input_path, category)
        zip_files = [f for f in os.listdir(category_path) if f.endswith('.zip')]
        print(f"  {category}: {len(zip_files)} zip files")
        total_zips += len(zip_files)
    
    print(f"Total zip files found: {total_zips}")
    return total_zips > 0

def process_your_dataset(input_path, output_path=None, store_in_db=True):
    """Process your specific dataset structure."""
    
    print("=" * 60)
    print("Personal RAG-LLM Dataset Processor")
    print("=" * 60)
    
    # Validate input
    if not validate_dataset_structure(input_path):
        print("❌ No zip files found in dataset structure")
        return False
    
    # Setup output directory
    if not output_path:
        output_path = os.path.join(os.path.dirname(input_path), "processed_results")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Create configuration
    config = create_config()
    
    # Initialize processor
    print("🔧 Initializing dataset processor...")
    processor = ZipDatasetProcessor(config)
    
    try:
        # Process the dataset
        print(f"📁 Processing dataset from: {input_path}")
        print(f"💾 Results will be saved to: {output_path}")
        
        document_pairs = processor.process_dataset(input_path, output_path)
        
        # Print detailed statistics
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
            success_rate = cat_stats['successful_pairs'] / cat_stats['zip_files'] if cat_stats['zip_files'] > 0 else 0
            print(f"  {category:15}: {cat_stats['successful_pairs']:3}/{cat_stats['zip_files']:3} ({success_rate:.1%})")
        
        if stats['errors']:
            print(f"\n⚠️  Errors encountered: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"    {error}")
            if len(stats['errors']) > 5:
                print(f"    ... and {len(stats['errors']) - 5} more errors")
        
        # Store in databases if requested
        if store_in_db:
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
                    for error in storage_results['storage_errors'][:3]:
                        print(f"    {error}")
                
            except Exception as e:
                print(f"❌ Database storage failed: {e}")
                print("Documents were processed but not stored in databases.")
        
        print(f"\n🎉 Processing complete! Check {output_path} for detailed results.")
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary files
        processor.cleanup()

def main():
    """Main entry point."""
    
    # Setup logging
    setup_logging()
    
    # Configuration - UPDATE THESE PATHS FOR YOUR SYSTEM
    INPUT_PATH = "/data/input"  # Path to your dataset with Documents, Images, etc. folders
    OUTPUT_PATH = "/data/processed_results"  # Where to save results
    STORE_IN_DB = True  # Whether to store in databases
    
    # You can also pass paths as command line arguments
    if len(sys.argv) > 1:
        INPUT_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_PATH = sys.argv[2]
    if len(sys.argv) > 3:
        STORE_IN_DB = sys.argv[3].lower() in ('true', '1', 'yes')
    
    print(f"Input path: {INPUT_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Store in database: {STORE_IN_DB}")
    
    # Process the dataset
    success = process_your_dataset(INPUT_PATH, OUTPUT_PATH, STORE_IN_DB)
    
    if success:
        print("\n✅ Dataset processing completed successfully!")
        
        # Print next steps
        print("\n📋 Next Steps:")
        print("1. Check the output directory for processing results")
        print("2. Review any failed documents and errors")
        print("3. Query your RAG system using the processed documents")
        print("4. Use the document relationships for enhanced retrieval")
        
    else:
        print("\n❌ Dataset processing failed!")
        print("Check the logs for error details.")
        sys.exit(1)

if __name__ == "__main__":
    main()