# services/processor/test_vlm_integration.py
#!/usr/bin/env python3
"""
Test script to verify VLM integration is working correctly with your document structure.
Tests the complete pipeline from ZIP file processing to JSON output.
"""

import os
import sys
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('src')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_vlm_integration():
    """Test the complete VLM integration pipeline."""
    
    print("🧪 VLM INTEGRATION TEST")
    print("="*50)
    
    # Step 1: Check prerequisites
    print("\n1. 🔍 Checking Prerequisites...")
    
    # Check if Ollama is running
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    print("   ✅ Ollama service is running")
                    
                    data = await response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    
                    if 'llava:7b' in models:
                        print("   ✅ LLaVA model is available")
                    else:
                        print("   ❌ LLaVA model not found")
                        print("   Run: ollama pull llava:7b")
                        return False
                else:
                    print(f"   ❌ Ollama responded with status {response.status}")
                    return False
    except Exception as e:
        print(f"   ❌ Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return False
    
    # Check input directory structure
    input_dir = "/Users/yashm/Documents/GitHub/SPEC-001_Personal-RAG-LLM/SPEC-001_Personal-RAG-LLM/data/input"
    print(f"\n2. 📂 Checking Input Directory Structure...")
    print(f"   Input directory: {input_dir}")
    
    if not os.path.exists(input_dir):
        print(f"   ❌ Input directory not found: {input_dir}")
        return False
    
    # Check for category directories
    expected_categories = ["Documents", "Emails", "Images", "Presentations", "Spreadsheets"]
    found_categories = []
    
    for category in expected_categories:
        category_path = os.path.join(input_dir, category)
        if os.path.exists(category_path):
            zip_files = [f for f in os.listdir(category_path) if f.endswith('.zip')]
            if zip_files:
                found_categories.append(category)
                print(f"   ✅ {category}: {len(zip_files)} ZIP files")
            else:
                print(f"   ⚠️ {category}: directory exists but no ZIP files")
        else:
            print(f"   ❌ {category}: directory not found")
    
    if not found_categories:
        print("   ❌ No categories with ZIP files found")
        return False
    
    print(f"   ✅ Found {len(found_categories)} categories with ZIP files")
    
    # Step 3: Test configuration loading
    print("\n3. ⚙️ Testing Configuration Loading...")
    
    try:
        from hydra import compose, initialize_config_dir
        
        config_dir = os.path.abspath("conf")
        
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(config_name="config", overrides=["vlm.enabled=true"])
            print("   ✅ Configuration loaded successfully")
            print(f"   VLM enabled: {config.vlm.enabled}")
            print(f"   VLM model: {config.vlm.llava.model}")
    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
        return False
    
    # Step 4: Test VLM processor initialization
    print("\n4. 🤖 Testing VLM Processor Initialization...")
    
    try:
        from pipeline.vlm_integrated_processor import VLMIntegratedProcessor
        
        config_dict = dict(config)
        processor = VLMIntegratedProcessor(config_dict)
        
        print("   ✅ VLM Integrated Processor initialized")
        print(f"   VLM enabled: {processor.vlm_processor.enabled}")
        print(f"   Temp directory: {processor.temp_dir}")
    except Exception as e:
        print(f"   ❌ VLM processor initialization failed: {e}")
        return False
    
    # Step 5: Test single ZIP file processing
    print("\n5. 📦 Testing Single ZIP File Processing...")
    
    # Find a test ZIP file
    test_zip = None
    test_category = None
    
    for category in found_categories:
        category_path = os.path.join(input_dir, category)
        zip_files = [f for f in os.listdir(category_path) if f.endswith('.zip')]
        if zip_files:
            test_zip = os.path.join(category_path, zip_files[0])
            test_category = category
            break
    
    if not test_zip:
        print("   ❌ No ZIP files found for testing")
        return False
    
    print(f"   Testing with: {os.path.basename(test_zip)} from {test_category}")
    
    try:
        # Create output directory for test
        test_output_dir = "./test_vlm_output"
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Process single ZIP file
        start_time = datetime.now()
        
        document_pair = await processor._process_zip_file(test_zip, test_category, test_output_dir)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   ✅ ZIP file processed in {processing_time:.1f}s")
        print(f"   Pair ID: {document_pair.pair_id}")
        print(f"   Base name: {document_pair.base_name}")
        
        # Check results
        if document_pair.content_doc:
            print(f"   📄 Content doc: {document_pair.content_doc.vlm_content_type} "
                  f"(confidence: {document_pair.content_doc.vlm_confidence:.2f})")
        
        if document_pair.metadata_doc:
            print(f"   📄 Metadata doc: {document_pair.metadata_doc.vlm_content_type} "
                  f"(confidence: {document_pair.metadata_doc.vlm_confidence:.2f})")
        
        if document_pair.errors:
            print(f"   ⚠️ Errors: {document_pair.errors}")
        
        # Save test results
        await processor._save_document_pair_results(document_pair, test_output_dir)
        
        print(f"   💾 Results saved to: {test_output_dir}/{document_pair.base_name}/")
        
        # Show JSON output structure
        pair_dir = os.path.join(test_output_dir, document_pair.base_name)
        if os.path.exists(pair_dir):
            json_files = [f for f in os.listdir(pair_dir) if f.endswith('.json')]
            print(f"   📁 JSON files created: {json_files}")
            
            # Show sample content from one JSON file
            if json_files:
                sample_file = os.path.join(pair_dir, json_files[0])
                with open(sample_file, 'r') as f:
                    sample_data = json.load(f)
                
                print(f"   📋 Sample JSON structure from {json_files[0]}:")
                print(f"      - doc_id: {sample_data.get('doc_id', 'N/A')}")
                print(f"      - vlm_content_type: {sample_data.get('vlm_content_type', 'N/A')}")
                print(f"      - vlm_confidence: {sample_data.get('vlm_confidence', 'N/A')}")
                print(f"      - processing_method: {sample_data.get('processing_method', 'N/A')}")
                print(f"      - text_content_length: {len(sample_data.get('text_content', ''))}")
        
    except Exception as e:
        print(f"   ❌ ZIP file processing failed: {e}")
        logger.exception("Full error details:")
        return False
    
    finally:
        # Cleanup
        processor.cleanup()
    
    # Step 6: Success summary
    print("\n6. 🎉 Test Summary")
    print("   ✅ All tests passed!")
    print("   ✅ VLM integration is working correctly")
    print("   ✅ JSON output is being generated")
    
    print(f"\n📁 Test output saved to: {test_output_dir}")
    print("🚀 Ready to process your full dataset!")
    
    return True

def print_next_steps():
    """Print next steps for the user."""
    
    print("\n" + "="*60)
    print("🎯 NEXT STEPS")
    print("="*60)
    
    print("\n📋 To process your full dataset:")
    print("# Test mode (one ZIP per category)")
    print("python3 process_with_vlm.py --test-mode")
    print("")
    print("# Process specific category")
    print("python3 process_with_vlm.py --category Documents --test-mode")
    print("")
    print("# Full processing (all categories)")
    print("python3 process_with_vlm.py")
    
    print("\n📊 To monitor progress:")
    print("- Watch the console output for real-time progress")
    print("- Check the log file for detailed information")
    print("- Results are saved as JSON files for each document pair")
    
    print("\n🔧 If you encounter issues:")
    print("- Check Ollama service: curl http://localhost:11434/api/tags")
    print("- Verify model: ollama list")
    print("- Increase timeouts if processing is slow")
    print("- Check logs for specific error messages")

async def main():
    """Main test function."""
    
    print("🧪 VLM INTEGRATION TEST SUITE")
    print("This will test your complete VLM document processing pipeline")
    print("")
    
    # Set VLM enabled
    os.environ["VLM_ENABLED"] = "true"
    
    success = await test_vlm_integration()
    
    if success:
        print_next_steps()
        print("\n✅ Integration test completed successfully!")
        return 0
    else:
        print("\n❌ Integration test failed. Please fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)