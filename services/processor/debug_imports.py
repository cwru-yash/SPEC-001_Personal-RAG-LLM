#!/usr/bin/env python3
"""Debug the imports in the VLM integrated processor."""

import sys
import os

print("🔍 Debugging VLM Integrated Processor Imports")
print("=" * 50)

# Add src to path
sys.path.append('src')

print(f"Current directory: {os.getcwd()}")
print(f"Python sys.path: {sys.path}")

# Test each import individually
imports_to_test = [
    "os",
    "sys", 
    "zipfile",
    "tempfile",
    "shutil",
    "json",
    "uuid",
    "asyncio",
    "logging",
    "pathlib.Path",
    "datetime.datetime",
    "concurrent.futures.ThreadPoolExecutor",
    "dataclasses.dataclass",
    "typing.Dict",
    "fitz",  # PyMuPDF
]

print("\n🧪 Testing standard library imports:")
for import_name in imports_to_test:
    try:
        if "." in import_name:
            module, item = import_name.rsplit(".", 1)
            exec(f"from {module} import {item}")
        else:
            exec(f"import {import_name}")
        print(f"   ✅ {import_name}")
    except ImportError as e:
        print(f"   ❌ {import_name}: {e}")
    except Exception as e:
        print(f"   ❌ {import_name}: {type(e).__name__}: {e}")

print("\n🤖 Testing VLM processor import:")

# Test the problematic import
cpu_vlm_import_attempts = [
    # Try different import paths
    "services.processor.src.pipeline.vlm.cpu_optimized_processor.CPUOptimizedVLMProcessor",
    "pipeline.vlm.cpu_optimized_processor.CPUOptimizedVLMProcessor", 
    "src.pipeline.vlm.cpu_optimized_processor.CPUOptimizedVLMProcessor",
]

for import_path in cpu_vlm_import_attempts:
    try:
        module_path, class_name = import_path.rsplit(".", 1)
        exec(f"from {module_path} import {class_name}")
        print(f"   ✅ SUCCESS: from {module_path} import {class_name}")
        working_import = import_path
        break
    except ImportError as e:
        print(f"   ❌ from {module_path} import {class_name}: {e}")
    except Exception as e:
        print(f"   ❌ from {module_path} import {class_name}: {type(e).__name__}: {e}")
else:
    print("   ❌ None of the import paths worked")
    working_import = None

print("\n📁 Checking file existence:")
vlm_paths_to_check = [
    "services/processor/src/pipeline/vlm/cpu_optimized_processor.py",
    "src/pipeline/vlm/cpu_optimized_processor.py",
    "pipeline/vlm/cpu_optimized_processor.py",
]

for path in vlm_paths_to_check:
    exists = os.path.exists(path)
    print(f"   {'✅' if exists else '❌'} {path}")

print("=" * 50)

if working_import:
    print(f"🎉 Use this import: from {working_import.rsplit('.', 1)[0]} import {working_import.rsplit('.', 1)[1]}")
else:
    print("💡 Need to fix the import path in the VLM integrated processor file")