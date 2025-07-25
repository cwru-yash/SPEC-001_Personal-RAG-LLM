# Create a debug version of your test script
# File: debug_vlm_test.py

import asyncio
import aiohttp
from src.pipeline.vlm.vlm_processor import VLMProcessor

async def debug_vlm_connection():
    """Debug VLM connection step by step."""
    
    print("=== VLM Connection Debug ===")
    
    # Step 1: Test basic connectivity
    print("1. Testing Ollama connectivity...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Ollama is running. Available models: {[m['name'] for m in data.get('models', [])]}")
                else:
                    print(f"❌ Ollama responded with status {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {str(e)}")
        print("   Make sure Ollama is running: 'ollama serve'")
        return False
    
    # Step 2: Test model availability
    print("\n2. Testing LLaVA model availability...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "llava:7b",
                "prompt": "Hello",
                "stream": False
            }
            async with session.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)  # Extended timeout for testing
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ LLaVA model responds: {result.get('response', '')[:50]}...")
                else:
                    error_text = await response.text()
                    print(f"❌ LLaVA model error {response.status}: {error_text}")
                    return False
    except asyncio.TimeoutError:
        print("❌ LLaVA model timeout - it's taking too long to respond")
        print("   This is normal on CPU. Consider increasing timeout or using GPU.")
        return False
    except Exception as e:
        print(f"❌ LLaVA model error: {str(e)}")
        return False
    
    print("✅ All connectivity tests passed!")
    return True

async def test_vlm_with_extended_timeouts():
    """Test VLM with more generous timeouts for CPU processing."""
    
    # Extended configuration for CPU processing
    config = {
        'vlm': {
            'enabled': True,
            'llava': {
                'model': 'llava:7b',
                'api_endpoint': 'http://localhost:11434',
                'timeout': 180,  # 3 minutes - more realistic for CPU
                'max_retries': 1  # Fewer retries to save time during testing
            },
            'strategy': 'vlm_first',
            'fallback': {
                'enabled': True,
                'timeout_threshold': 150,  # More generous threshold
                'confidence_threshold': 0.3  # Lower confidence threshold
            },
            'prompts': {
                'universal': '''Analyze this document page briefly.
                
                Respond in JSON format:
                {
                  "content_type": "report",
                  "confidence": 0.8,
                  "layout": "single column text",
                  "text_content": "extracted text here",
                  "visual_elements": [],
                  "metadata": {}
                }'''  # Shorter prompt to reduce processing time
            }
        }
    }
    
    print("\n=== Testing VLM Processing ===")
    
    # Test connectivity first
    if not await debug_vlm_connection():
        print("❌ Connectivity tests failed. Fix connectivity issues first.")
        return
    
    print("\n3. Testing VLM document processing with extended timeouts...")
    
    processor = VLMProcessor(config)
    
    # Test with a simple PDF (limit to first page only for testing)
    try:
        # You can modify the VLM processor to limit pages for testing
        result = await processor.process_document(
            '/Users/yashm/Documents/GitHub/SPEC-001_Personal-RAG-LLM/SPEC-001_Personal-RAG-LLM/data/input/Documents/frdx0256/frdx0256.pdf'
        )
        
        print(f"✅ VLM processing completed!")
        print(f"   Processing method: {result.get('processing_method', 'unknown')}")
        print(f"   Pages processed: {len(result.get('pages', []))}")
        
        # Show results from first page
        if result.get('pages'):
            first_page = result['pages'][0]
            vlm_result = first_page.get('vlm_result', {})
            print(f"   First page success: {vlm_result.get('success', False)}")
            print(f"   Content type detected: {vlm_result.get('content_type', 'unknown')}")
            print(f"   Confidence: {vlm_result.get('confidence', 0.0):.2f}")
            
    except Exception as e:
        print(f"❌ VLM processing failed: {str(e)}")

# Run the debug tests
if __name__ == "__main__":
    asyncio.run(test_vlm_with_extended_timeouts())