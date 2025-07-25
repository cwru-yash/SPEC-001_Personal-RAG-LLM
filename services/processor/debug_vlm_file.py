#!/usr/bin/env python3
"""
Comprehensive debugging for the VLM integrated processor file.
This will help us find exactly what's preventing the module from loading.
"""

import sys
import os

print("🔍 COMPREHENSIVE VLM FILE DEBUGGING")
print("=" * 60)

# Add src to path
sys.path.append('src')

# Step 1: Check if the class is actually defined in the file
print("\n1. 📝 Checking if Qwen25VLProcessor class is defined...")

vlm_file_path = "src/pipeline/routing/document_router.py"

try:
    with open(vlm_file_path, 'r') as f:
        file_content = f.read()
    
    if "class Qwen25VLProcessor" in file_content:
        print("   ✅ Qwen25VLProcessor class found in file")
        
        # Find the line number where the class is defined
        lines = file_content.split('\n')
        for i, line in enumerate(lines, 1):
            if "class Qwen25VLProcessor" in line:
                print(f"   📍 Class defined at line {i}: {line.strip()}")
                break
    else:
        print("   ❌ Qwen25VLProcessor class NOT found in file")
        print("   💡 This is likely the problem - the class definition is missing")
        
        # Check what classes are defined
        lines = file_content.split('\n')
        found_classes = []
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("class "):
                found_classes.append(f"Line {i}: {line.strip()}")
        
        if found_classes:
            print("   📋 Classes found in file:")
            for class_def in found_classes:
                print(f"      {class_def}")
        else:
            print("   ❌ No class definitions found in file")

except Exception as e:
    print(f"   ❌ Error reading file: {e}")
    sys.exit(1)

# Step 2: Check imports within the file
print("\n2. 🔗 Testing imports from within the VLM file...")

# Extract import statements from the file
import_lines = []
lines = file_content.split('\n')

for line in lines:
    stripped = line.strip()
    if stripped.startswith('import ') or stripped.startswith('from '):
        import_lines.append(stripped)

print(f"   Found {len(import_lines)} import statements:")

for i, import_line in enumerate(import_lines, 1):
    print(f"   {i:2d}. {import_line}")
    
    # Test each import
    try:
        exec(import_line)
        print(f"       ✅ Success")
    except ImportError as e:
        print(f"       ❌ ImportError: {e}")
    except Exception as e:
        print(f"       ❌ {type(e).__name__}: {e}")

# Step 3: Try to execute the file directly to see what fails
print("\n3. 🚀 Attempting to execute the file directly...")

try:
    # This will show us exactly where the problem occurs
    exec(compile(file_content, vlm_file_path, 'exec'))
    print("   ✅ File executed successfully")
    print("   💡 The problem might be with the import path, not the file content")
    
except SyntaxError as e:
    print(f"   ❌ Syntax Error at line {e.lineno}: {e.msg}")
    print(f"       Problem text: {e.text}")
    
except ImportError as e:
    print(f"   ❌ Import Error: {e}")
    print("   💡 One of the imports in the file is failing")
    
except Exception as e:
    print(f"   ❌ Runtime Error: {type(e).__name__}: {e}")
    import traceback
    print("   📜 Full traceback:")
    traceback.print_exc()

# Step 4: Check if the problematic dependency exists
print("\n4. 🎯 Testing the specific VLM processor dependency...")

cpu_vlm_paths = [
    "src/pipeline/vlm/cpu_optimized_processor.py",
    "pipeline/vlm/cpu_optimized_processor.py"
]

for path in cpu_vlm_paths:
    exists = os.path.exists(path)
    print(f"   {'✅' if exists else '❌'} {path}")

# Try the import that's likely causing issues
print("\n   Testing CPUOptimizedVLMProcessor import:")
try:
    from pipeline.vlm.cpu_optimized_processor import CPUOptimizedVLMProcessor
    print("   ✅ CPUOptimizedVLMProcessor imports successfully")
except ImportError as e:
    print(f"   ❌ CPUOptimizedVLMProcessor import failed: {e}")
except Exception as e:
    print(f"   ❌ CPUOptimizedVLMProcessor import error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("🎯 DIAGNOSIS SUMMARY:")
print("If the class definition is missing, that's your main problem.")
print("If an import is failing, that's preventing the module from loading.")
print("If there's a syntax error, that needs to be fixed first.")