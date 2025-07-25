#!/usr/bin/env python3
"""
Run the enhanced dataset processor with your files.
Usage: python run_processor.py [input_path] [output_path] [--test-single-category CATEGORY]
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def create_config():
    """Create processing configuration."""
    return {
        # Processing settings
        "max_workers": 4,
        "max_file_size_mb": 200,
        
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
            
            # Text processing
            "text": {
                "encoding_fallbacks": ["utf-8", "latin-1", "cp1252"],
                "csv": {
                    "max_preview_rows": 100
                }
            },
            
            # Image processing
            "image": {
                "ocr": {
                    "engine": "tesseract",
                    "languages": "eng",
                    "preprocess": True
                }
            },
            
            # Chunking
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
        
        # Storage configuration (Docker)
        "storage": {
            "duckdb": {
                "database": "/data/metadata.db"
            },
            # "vector_store": {
            #     "host": "localhost",  # or "chroma" if running in Docker
            #     "port": 8000
            # }
            "vector_store": {
                        "host": "chroma",  # Not "localhost" when running in Docker
                        "port": 8000
                    },
            "graph_db": {
                "uri": "bolt://localhost:7687",  # or "neo4j:7687" if running in Docker
                "user": "neo4j",
                "password": "password",
                "database": "neo4j"
            }
        }
    }

def test_single_category(input_path, category, output_path):
    """Test processing for a single category."""
    from services.processor.src.pipeline.standalone_dataset_processor import EnhancedDatasetProcessor
    
    print(f"🧪 Testing category: {category}")
    
    category_path = os.path.join(input_path, category)
    if not os.path.exists(category_path):
        print(f"❌ Category not found: {category_path}")
        return False
    
    # Get first few ZIP files for testing
    zip_files = [f for f in os.listdir(category_path) if f.endswith('.zip')][:3]
    
    if not zip_files:
        print(f"❌ No ZIP files found in {category}")
        return False
    
    print(f"Testing with {len(zip_files)} ZIP files: {zip_files}")
    
    # Create temporary test structure
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test input structure
        test_input = os.path.join(temp_dir, "test_input")
        test_category_dir = os.path.join(test_input, category)
        os.makedirs(test_category_dir)
        
        # Copy test files
        for zip_file in zip_files:
            src = os.path.join(category_path, zip_file)
            dst = os.path.join(test_category_dir, zip_file)
            shutil.copy2(src, dst)
        
        # Process
        test_output = os.path.join(temp_dir, "test_output")
        config = create_config()
        processor = EnhancedDatasetProcessor(config["max_workers"])
        
        try:
            pairs = processor.process_dataset(test_input, test_output)
            
            print(f"✅ Processed {len(pairs)} document pairs")
            
            # Show results
            successful = len([p for p in pairs if p.content_doc and p.metadata_doc])
            print(f"   Successful pairs: {successful}/{len(pairs)}")
            
            # Copy results to permanent location
            if os.path.exists(test_output):
                final_output = os.path.join(output_path, f"test_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                shutil.copytree(test_output, final_output)
                print(f"   Results saved to: {final_output}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            processor.cleanup()

def process_full_dataset(input_path, output_path):
    """Process the full dataset."""
    from services.processor.src.pipeline.standalone_dataset_processor import EnhancedDatasetProcessor
    
    print("🚀 Processing Full Dataset")
    print("=" * 50)
    
    # Validate input
    if not os.path.exists(input_path):
        print(f"❌ Input path not found: {input_path}")
        return False
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize processor
    config = create_config()
    processor = EnhancedDatasetProcessor(config["max_workers"])
    
    try:
        # Process dataset
        pairs = processor.process_dataset(input_path, output_path)
        
        # Print results
        print(f"\n🎉 Processing Complete!")
        print(f"Total document pairs: {len(pairs)}")
        
        successful = len([p for p in pairs if p.content_doc and p.metadata_doc])
        print(f"Successful pairs: {successful}")
        print(f"Success rate: {successful/len(pairs)*100:.1f}%" if pairs else "0%")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        processor.cleanup()

def store_in_databases(pairs, config):
    """Store processed documents in databases."""
    print("\n💾 Storing in Databases...")
    
    try:
        # Import storage classes
        from services.processor.src.storage.duckdb import DuckDBStorage
        from services.processor.src.storage.vector_store import VectorStore
        from services.processor.src.storage.graph_db import GraphDBStorage
        from services.processor.src.pipeline.zip_dataset_processor import ZipDatasetStorage
        
        # Initialize storage
        metadata_storage = DuckDBStorage(config["storage"]["duckdb"]["database"])
        vector_store = VectorStore(config["storage"]["vector_store"])
        graph_storage = GraphDBStorage(config["storage"]["graph_db"])
        
        # Store documents
        storage = ZipDatasetStorage(metadata_storage, vector_store, graph_storage)
        results = storage.store_document_pairs(pairs)
        
        print(f"✅ Stored {results['stored_content_docs']} content documents")
        print(f"✅ Stored {results['stored_metadata_docs']} metadata documents")
        print(f"✅ Created {results['relationships_created']} relationships")
        
        if results['storage_errors']:
            print(f"⚠️  {len(results['storage_errors'])} storage errors")
            for error in results['storage_errors'][:3]:
                print(f"    {error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database storage failed: {e}")
        print("Documents were processed but not stored in databases.")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point."""

    """Main test function."""
        logger = setup_logging()
        config = create_test_config()
        
        # Only run the zip processor test
        test_results = {}
        test_results["zip_processor"] = test_enhanced_zip_processor(config, logger)
        
        generate_test_report(test_results, logger)

    # setup_logging()
    
    # Parse arguments
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/data/input"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/data/processed_results"
    
    # Check for test mode
    if "--test-single-category" in sys.argv:
        category_idx = sys.argv.index("--test-single-category") + 1
        if category_idx < len(sys.argv):
            category = sys.argv[category_idx]
            return test_single_category(input_path, category, output_path)
        else:
            print("❌ Please specify category after --test-single-category")
            return False
    
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    missing_deps = []
    
    try:
        import fitz
        print("✅ PyMuPDF available")
    except ImportError:
        missing_deps.append("pymupdf")
    
    try:
        import openpyxl
        print("✅ openpyxl available")
    except ImportError:
        missing_deps.append("openpyxl")
    
    try:
        import pytesseract
        print("✅ pytesseract available")
    except ImportError:
        missing_deps.append("pytesseract")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        return False
    
    # Process dataset
    success = process_full_dataset(input_path, output_path)
    
    if success:
        print("\n✅ Processing completed successfully!")
        print(f"📁 Results available in: {output_path}")
        print("\n📋 Next Steps:")
        print("1. Review processing results in the output directory")
        print("2. Check logs for any errors or warnings")
        print("3. Test with database storage if services are running")
        print("4. Query your processed documents")
    else:
        print("\n❌ Processing failed!")
        print("Check the logs for error details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)