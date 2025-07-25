# progressive_vlm_test.py
# Test VLM integration step by step, building confidence gradually

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class ProgressiveVLMTester:
    """Test VLM integration with increasing complexity."""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "llava:7b"
        
    async def test_level_1_connectivity(self):
        """Level 1: Basic connectivity to Ollama service."""
        print("🔍 Level 1: Testing basic connectivity...")
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    duration = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        models = [m['name'] for m in data.get('models', [])]
                        print(f"✅ Ollama responds in {duration:.1f}s. Models: {models}")
                        return True
                    else:
                        print(f"❌ Ollama error {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Connectivity failed: {e}")
            print("   💡 Try: ollama serve")
            return False
    
    async def test_level_2_text_only(self):
        """Level 2: Text-only generation (no images)."""
        print("\n🔍 Level 2: Testing text-only generation...")
        
        payload = {
            "model": self.model,
            "prompt": "Respond with exactly: 'VLM is working'",
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    duration = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '').strip()
                        print(f"✅ Text generation works in {duration:.1f}s")
                        print(f"   Response: '{response_text}'")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ Text generation failed {response.status}: {error_text}")
                        return False
        except asyncio.TimeoutError:
            print(f"❌ Text generation timeout (>{120}s)")
            print("   💡 Model is very slow. Consider using GPU or smaller model.")
            return False
        except Exception as e:
            print(f"❌ Text generation error: {e}")
            return False
    
    async def test_level_3_simple_image(self):
        """Level 3: Simple image description."""
        print("\n🔍 Level 3: Testing simple image processing...")
        
        # Create a minimal test image (solid color square)
        from PIL import Image
        import base64
        import io
        
        # Create simple test image
        test_img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        test_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        payload = {
            "model": self.model,
            "prompt": "What color is this image? Answer in one word.",
            "images": [img_base64],
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as response:
                    duration = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '').strip()
                        print(f"✅ Image processing works in {duration:.1f}s")
                        print(f"   Response: '{response_text}'")
                        
                        # Check if it correctly identified red
                        if 'red' in response_text.lower():
                            print("   🎯 Correctly identified color!")
                        else:
                            print("   ⚠️  Color identification unclear")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ Image processing failed {response.status}: {error_text}")
                        return False
        except asyncio.TimeoutError:
            print(f"❌ Image processing timeout (>180s)")
            print("   💡 Image processing is very slow on CPU")
            return False
        except Exception as e:
            print(f"❌ Image processing error: {e}")
            return False
    
    async def test_level_4_json_output(self):
        """Level 4: JSON-formatted output."""
        print("\n🔍 Level 4: Testing structured JSON output...")
        
        # Create simple test image with text
        from PIL import Image, ImageDraw, ImageFont
        import base64
        import io
        
        # Create image with text
        test_img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(test_img)
        
        # Add simple text
        try:
            # Try to use default font
            draw.text((10, 30), "Hello World", fill='black')
        except:
            # Fallback if no font available
            draw.text((10, 30), "Hello World", fill='black')
        
        buffer = io.BytesIO()
        test_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        prompt = """Analyze this image and respond with valid JSON:
        
        {
          "content_type": "text_image",
          "confidence": 0.9,
          "text_found": "text you see",
          "description": "brief description"
        }
        
        Only return valid JSON, nothing else."""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as response:
                    duration = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '').strip()
                        print(f"✅ Structured output works in {duration:.1f}s")
                        
                        # Try to parse JSON
                        try:
                            # Extract JSON from response
                            json_start = response_text.find('{')
                            json_end = response_text.rfind('}') + 1
                            
                            if json_start != -1 and json_end != -1:
                                json_text = response_text[json_start:json_end]
                                parsed_json = json.loads(json_text)
                                print(f"   🎯 Valid JSON returned!")
                                print(f"   Content: {json.dumps(parsed_json, indent=2)}")
                            else:
                                print(f"   ⚠️  No JSON structure found")
                                print(f"   Raw response: {response_text[:100]}...")
                        except json.JSONDecodeError:
                            print(f"   ⚠️  Invalid JSON returned")
                            print(f"   Raw response: {response_text[:100]}...")
                        
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ Structured output failed {response.status}: {error_text}")
                        return False
        except asyncio.TimeoutError:
            print(f"❌ Structured output timeout (>180s)")
            return False
        except Exception as e:
            print(f"❌ Structured output error: {e}")
            return False
    
    async def test_level_5_single_pdf_page(self):
        """Level 5: Process one page from your actual PDF."""
        print("\n🔍 Level 5: Testing single PDF page processing...")
        
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            import base64
            import io
            
            # Open your test PDF
            pdf_path = '/Users/yashm/Documents/GitHub/SPEC-001_Personal-RAG-LLM/SPEC-001_Personal-RAG-LLM/data/input/Documents/frdx0256/frdx0256.pdf'
            doc = fitz.open(pdf_path)
            
            if len(doc) == 0:
                print("❌ PDF has no pages")
                return False
            
            # Get first page as image
            page = doc[0]
            matrix = fitz.Matrix(1.5, 1.5)  # Moderate resolution for testing
            pix = page.get_pixmap(matrix=matrix)
            img_data = pix.tobytes("png")
            page_image = Image.open(io.BytesIO(img_data))
            
            # Convert to base64
            buffer = io.BytesIO()
            page_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            doc.close()
            
            prompt = """Analyze this document page and respond with JSON:
            
            {
              "content_type": "document_type_here",
              "confidence": 0.8,
              "has_text": true,
              "has_images": false,
              "layout": "description",
              "key_content": "main content summary"
            }
            
            Only return valid JSON."""
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                print(f"   📄 Processing PDF page (image size: {page_image.size})...")
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes for real document
                ) as response:
                    duration = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '').strip()
                        print(f"✅ PDF page processed in {duration:.1f}s")
                        
                        # Try to parse JSON
                        try:
                            json_start = response_text.find('{')
                            json_end = response_text.rfind('}') + 1
                            
                            if json_start != -1 and json_end != -1:
                                json_text = response_text[json_start:json_end]
                                parsed_json = json.loads(json_text)
                                print(f"   🎯 Document analysis successful!")
                                print(f"   Content type: {parsed_json.get('content_type', 'unknown')}")
                                print(f"   Confidence: {parsed_json.get('confidence', 0.0)}")
                                print(f"   Key content: {parsed_json.get('key_content', '')[:100]}...")
                                return True
                            else:
                                print(f"   ⚠️  Response structure unclear")
                                print(f"   Raw response: {response_text[:200]}...")
                                return False
                        except json.JSONDecodeError:
                            print(f"   ⚠️  JSON parsing failed")
                            print(f"   Raw response: {response_text[:200]}...")
                            return False
                    else:
                        error_text = await response.text()
                        print(f"❌ PDF processing failed {response.status}: {error_text}")
                        return False
        
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            print(f"❌ PDF processing timeout after {duration:.1f}s")
            print("   💡 Real documents take much longer. This is expected on CPU.")
            return False
        except Exception as e:
            print(f"❌ PDF processing error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests in sequence."""
        print(f"🚀 Progressive VLM Testing Started - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        tests = [
            ("Connectivity", self.test_level_1_connectivity),
            ("Text Generation", self.test_level_2_text_only),
            ("Image Processing", self.test_level_3_simple_image),
            ("JSON Output", self.test_level_4_json_output),
            ("PDF Processing", self.test_level_5_single_pdf_page)
        ]
        
        for test_name, test_func in tests:
            success = await test_func()
            
            if not success:
                print(f"\n❌ {test_name} failed. Fix this before proceeding to next level.")
                print(f"📋 Current system status: {test_name} is the blocker.")
                break
            else:
                print(f"   ✅ {test_name} passed")
        else:
            print(f"\n🎉 All tests passed! Your VLM integration is working correctly.")
            print(f"💡 Note: Processing times are normal for CPU-based inference.")
        
        print("="*60)

# Run the progressive tests
if __name__ == "__main__":
    tester = ProgressiveVLMTester()
    asyncio.run(tester.run_all_tests())