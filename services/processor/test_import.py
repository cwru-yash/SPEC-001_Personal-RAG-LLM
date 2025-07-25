#!/usr/bin/env python3
"""Quick test to verify the VLM integrated processor can be imported."""

import sys
import os

print("🔍 Testing VLM Integrated Processor Import")
print("=" * 50)

# Add src to path (same as the test script does)
sys.path.append('src')

print(f"Current directory: {os.getcwd()}")
print(f"Python path includes: {sys.path[-1]}")

# Check if file exists
vlm_processor_path = "src/pipeline/vlm_integrated_processor.py"
print(f"Checking file: {vlm_processor_path}")

if os.path.exists(vlm_processor_path):
    print("✅ File exists")
    
    # Check file size
    size = os.path.getsize(vlm_processor_path)
    print(f"✅ File size: {size} bytes")
    
    if size < 100:
        print("❌ File appears to be empty or nearly empty")
        print("   Make sure you copied the content from the artifact")
    else:
        print("✅ File has content")
        
        # Try to import
        try:
            print("🔄 Attempting import...")
            from pipeline.vlm_integrated_processor import VLMIntegratedProcessor
            print("✅ Import successful!")
            print(f"✅ Class loaded: {VLMIntegratedProcessor}")
            
        except SyntaxError as e:
            print(f"❌ Syntax error in file: {e}")
            print("   Check the file for syntax issues")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("   Check for missing dependencies or import issues within the file")
            
        except Exception as e:
            print(f"❌ Other error: {e}")
            
else:
    print("❌ File does not exist")
    print("   Create the file and copy the content from the artifact")

print("\n" + "=" * 50)