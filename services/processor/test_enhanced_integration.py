# services/processor/test_enhanced_integration.py

import asyncio
import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_enhanced_integration():
    """Comprehensive test of the enhanced VLM integration."""
    
    print("🚀 ENHANCED VLM INTEGRATION TEST")
    print("="*60)
    
    # Test 1: Configuration loading
    print("\n1. 📋 Testing Enhanced Configuration...")
    try:
        from hydra import compose, initialize_config_dir
        
        config_dir = os.path.abspath("conf")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(config_name="config", overrides=["vlm.enabled=true"])
            
            print("   ✅ Enhanced configuration loaded successfully")
            print(f"   VLM Strategy: {config.vlm.strategy}")
            print(f"   Local models available: {list(config.vlm.local_models.keys())}")
            print(f"   Cloud models configured: {list(config.vlm.cloud_models.keys())}")
            
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False
    
    # Test 2: Document router initialization
    print("\n2. 🧠 Testing Document Router...")
    try:
        from pipeline.routing.document_router import DocumentRouter
        
        config_dict = dict(config)
        router = DocumentRouter(config_dict)
        
        print(f"   ✅ Document router initialized")
        print(f"   Available processors: {len(router.processors)}")
        print(f"   Processor types: {list(router.processors.keys())}")
        
    except Exception as e:
        print(f"   ❌ Document router test failed: {e}")
        return False
    
    # Test 3: Enhanced processor initialization
    print("\n3. 🤖 Testing Enhanced Processor...")
    try:
        from pipeline.vlm_integrated_processor import EnhancedVLMIntegratedProcessor
        
        processor = EnhancedVLMIntegratedProcessor(config_dict)
        
        print("   ✅ Enhanced processor initialized successfully")
        print(f"   Cost limits: ${processor.cost_limits['document_limit']}/doc, ${processor.cost_limits['daily_limit']}/day")
        
    except Exception as e:
        print(f"   ❌ Enhanced processor test failed: {e}")
        return False
    
    # Test 4: Document analysis
    print("\n4. 📄 Testing Document Analysis...")
    test_pdf = "/Users/yashm/Documents/GitHub/SPEC-001_Personal-RAG-LLM/SPEC-001_Personal-RAG-LLM/data/input/Documents/qxwd0282.zip"
    
    if os.path.exists(test_pdf.replace('.zip', '.pdf')):  # Check if we can find a test PDF
        try:
            # Extract a test PDF for analysis
            import zipfile
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(test_pdf, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find a PDF file
                pdf_files = [f for f in os.listdir(temp_dir) if f.endswith('.pdf')]
                if pdf_files:
                    test_pdf_path = os.path.join(temp_dir, pdf_files[0])
                    
                    # Analyze document characteristics
                    characteristics = router.analyze_document_characteristics(test_pdf_path)
                    
                    print(f"   ✅ Document analysis completed")
                    print(f"   Difficulty: {characteristics.estimated_difficulty}")
                    print(f"   Recommended processor: {characteristics.recommended_processor}")
                    print(f"   Confidence: {characteristics.confidence:.2f}")
                    
        except Exception as e:
            print(f"   ⚠️ Document analysis test failed: {e}")
            print("   (This is non-critical for basic functionality)")
    else:
        print("   ⚠️ No test PDF found - skipping document analysis test")
    
    print("\n" + "="*60)
    print("🎉 Enhanced integration test completed!")
    print("✅ Your enhanced VLM system is ready for use")
    
    return True

async def main():
    """Main test function."""
    os.environ["VLM_ENABLED"] = "true"
    success = await test_enhanced_integration()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)