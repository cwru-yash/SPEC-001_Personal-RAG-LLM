# # inspect_dataset.py - Test and validation script for your dataset
# import os
# import sys
# import zipfile
# import tempfile
# import shutil
# from pathlib import Path

# # Add the project root to Python path
# sys.path.insert(0, os.path.abspath('.'))

# def inspect_dataset_structure(input_path):
#     """Inspect and validate the dataset structure."""
    
#     print("🔍 Dataset Structure Inspector")
#     print("=" * 50)
    
#     if not os.path.exists(input_path):
#         print(f"❌ Input path does not exist: {input_path}")
#         return False
    
#     print(f"📁 Inspecting: {input_path}")
    
#     # Get all directories (categories)
#     categories = []
#     for item in os.listdir(input_path):
#         item_path = os.path.join(input_path, item)
#         if os.path.isdir(item_path) and not item.startswith('.'):
#             categories.append(item)
    
#     if not categories:
#         print("❌ No category directories found")
#         return False
    
#     print(f"\n📂 Found {len(categories)} categories:")
    
#     total_zips = 0
#     category_details = {}
    
#     for category in sorted(categories):
#         category_path = os.path.join(input_path, category)
        
#         # Find zip files
#         zip_files = []
#         other_files = []
        
#         for file in os.listdir(category_path):
#             file_path = os.path.join(category_path, file)
#             if os.path.isfile(file_path):
#                 if file.endswith('.zip'):
#                     zip_files.append(file)
#                 else:
#                     other_files.append(file)
        
#         category_details[category] = {
#             'zip_files': zip_files,
#             'other_files': other_files,
#             'zip_count': len(zip_files)
#         }
        
#         total_zips += len(zip_files)
        
#         print(f"  📁 {category:15}: {len(zip_files):3} zip files")
#         if other_files:
#             print(f"     └─ Also contains: {len(other_files)} other files")
    
#     print(f"\n📊 Total: {total_zips} zip files across {len(categories)} categories")
    
#     return category_details

# def inspect_sample_zip_files(input_path, category_details, samples_per_category=2):
#     """Inspect sample zip files to understand their structure."""
    
#     print(f"\n🔬 Inspecting Sample ZIP Files ({samples_per_category} per category)")
#     print("=" * 60)
    
#     temp_dir = tempfile.mkdtemp(prefix="dataset_inspect_")
    
#     try:
#         for category, details in category_details.items():
#             if not details['zip_files']:
#                 continue
                
#             print(f"\n📂 Category: {category}")
#             print("-" * 30)
            
#             # Sample first few zip files
#             sample_zips = details['zip_files'][:samples_per_category]
            
#             for zip_name in sample_zips:
#                 zip_path = os.path.join(input_path, category, zip_name)
#                 print(f"\n  📦 {zip_name}")
                
#                 try:
#                     inspect_single_zip(zip_path, temp_dir)
#                 except Exception as e:
#                     print(f"    ❌ Error inspecting {zip_name}: {e}")
                    
#     finally:
#         # Clean up
#         if os.path.exists(temp_dir):
#             shutil.rmtree(temp_dir)

# def inspect_single_zip(zip_path, temp_dir):
#     """Inspect a single zip file."""
    
#     zip_name = os.path.basename(zip_path)
#     base_name = os.path.splitext(zip_name)[0]
    
#     # Extract to temporary directory
#     extract_path = os.path.join(temp_dir, base_name)
#     os.makedirs(extract_path, exist_ok=True)
    
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         file_list = zip_ref.namelist()
#         print(f"    📄 Contains {len(file_list)} files:")
        
#         for file in file_list:
#             file_size = zip_ref.getinfo(file).file_size
#             print(f"      - {file} ({file_size:,} bytes)")
        
#         # Extract files
#         zip_ref.extractall(extract_path)
    
#     # Find PDF files
#     pdf_files = []
#     for root, _, files in os.walk(extract_path):
#         for file in files:
#             if file.lower().endswith('.pdf'):
#                 pdf_files.append(os.path.join(root, file))
    
#     if len(pdf_files) == 2:
#         # Try to identify metadata vs content
#         pdf_info = []
#         for pdf_path in pdf_files:
#             pdf_name = os.path.basename(pdf_path)
#             pdf_size = os.path.getsize(pdf_path)
            
#             # Determine type
#             if 'info' in pdf_name.lower() or pdf_name.endswith('-info.pdf'):
#                 pdf_type = "📝 Metadata"
#             else:
#                 pdf_type = "📄 Content"
            
#             pdf_info.append((pdf_type, pdf_name, pdf_size))
        
#         # Sort by size (metadata usually smaller)
#         pdf_info.sort(key=lambda x: x[2])
        
#         print(f"    🔍 PDF Analysis:")
#         for pdf_type, pdf_name, pdf_size in pdf_info:
#             print(f"      {pdf_type:12}: {pdf_name} ({pdf_size:,} bytes)")
            
#         # Quick content peek at metadata PDF (smaller one)
#         metadata_pdf = pdf_info[0][1]
#         metadata_path = None
#         for pdf_path in pdf_files:
#             if os.path.basename(pdf_path) == metadata_pdf:
#                 metadata_path = pdf_path
#                 break
        
#         if metadata_path:
#             try:
#                 peek_pdf_content(metadata_path)
#             except Exception as e:
#                 print(f"    ⚠️  Could not peek at PDF content: {e}")
                
#     else:
#         print(f"    ⚠️  Expected 2 PDFs, found {len(pdf_files)}")
    
#     # Clean up
#     shutil.rmtree(extract_path)

# def peek_pdf_content(pdf_path, max_chars=200):
#     """Peek at PDF content to understand structure."""
    
#     try:
#         import fitz  # PyMuPDF
        
#         with fitz.open(pdf_path) as doc:
#             if len(doc) > 0:
#                 first_page = doc[0]
#                 text = first_page.get_text()
                
#                 if text.strip():
#                     # Show first few lines
#                     lines = text.strip().split('\n')[:5]
#                     preview = '\n'.join(lines)
                    
#                     if len(preview) > max_chars:
#                         preview = preview[:max_chars] + "..."
                    
#                     print(f"    👀 Content Preview:")
#                     for line in preview.split('\n'):
#                         if line.strip():
#                             print(f"      | {line.strip()}")
#                 else:
#                     print(f"    📷 Content: Image-only PDF (no text)")
            
#     except ImportError:
#         print(f"    ⚠️  PyMuPDF not available for content preview")
#     except Exception as e:
#         print(f"    ⚠️  Error reading PDF: {e}")

# def test_processing_setup():
#     """Test if the processing pipeline can be imported and initialized."""
    
#     print("\n🧪 Testing Processing Setup")
#     print("=" * 40)
    
#     try:
#         from services.processor.src.pipeline.zip_dataset_processor import ZipDatasetProcessor
#         print("✅ ZipDatasetProcessor imported successfully")
        
#         # Test basic initialization
#         config = {
#             "max_workers": 2,
#             "pipeline": {
#                 "pdf": {"engine": "pymupdf"},
#                 "chunker": {"default_chunk_size": 500}
#             }
#         }
        
#         processor = ZipDatasetProcessor(config)
#         print("✅ ZipDatasetProcessor initialized successfully")
        
#         # Test cleanup
#         processor.cleanup()
#         print("✅ Cleanup works correctly")
        
#         return True
        
#     except ImportError as e:
#         print(f"❌ Import error: {e}")
#         print("   Make sure all dependencies are installed:")
#         print("   pip install pymupdf pytesseract pillow")
#         return False
        
#     except Exception as e:
#         print(f"❌ Setup error: {e}")
#         return False

# def generate_processing_command(input_path, output_path=None):
#     """Generate the command to process the dataset."""
    
#     if not output_path:
#         output_path = os.path.join(os.path.dirname(input_path), "processed_results")
    
#     print(f"\n🚀 Ready to Process Dataset")
#     print("=" * 40)
#     print(f"To process your dataset, run:")
#     print(f"")
#     print(f"python process_dataset.py '{input_path}' '{output_path}'")
#     print(f"")
#     print(f"Or edit the paths in process_dataset.py and run:")
#     print(f"python process_dataset.py")

# def main():
#     """Main inspection function."""
    
#     print("📋 Personal RAG-LLM Dataset Inspector")
#     print("=" * 50)
    
#     # Get input path
#     if len(sys.argv) > 1:
#         input_path = sys.argv[1]
#     else:
#         # Default path - update this to your dataset location
#         input_path = "/data/input"
#         print(f"Using default input path: {input_path}")
#         print("(Pass a different path as argument: python inspect_dataset.py /path/to/dataset)")
    
#     # Step 1: Inspect dataset structure
#     category_details = inspect_dataset_structure(input_path)
    
#     if not category_details:
#         print("\n❌ Dataset inspection failed")
#         return
    
#     # Step 2: Inspect sample files
#     if any(details['zip_count'] > 0 for details in category_details.values()):
#         inspect_sample_zip_files(input_path, category_details)
#     else:
#         print("\n⚠️  No zip files found to inspect")
#         return
    
#     # Step 3: Test processing setup
#     setup_ok = test_processing_setup()
    
#     # Step 4: Generate processing command
#     if setup_ok:
#         generate_processing_command(input_path)
        
#         print(f"\n✅ Dataset is ready for processing!")
#         print(f"📊 Found {sum(d['zip_count'] for d in category_details.values())} zip files")
#         print(f"📂 Categories: {', '.join(category_details.keys())}")
#     else:
#         print(f"\n❌ Setup issues need to be resolved before processing")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# inspect_dataset_fixed.py - Fixed version that handles your specific dataset structure

import os
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path

def inspect_dataset_structure(input_path):
    """Inspect and validate the dataset structure."""
    
    print("🔍 Dataset Structure Inspector")
    print("=" * 50)
    
    if not os.path.exists(input_path):
        print(f"❌ Input path does not exist: {input_path}")
        return False
    
    print(f"📁 Inspecting: {input_path}")
    
    # Get all directories (categories)
    categories = []
    for item in os.listdir(input_path):
        item_path = os.path.join(input_path, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            categories.append(item)
    
    if not categories:
        print("❌ No category directories found")
        return False
    
    print(f"\n📂 Found {len(categories)} categories:")
    
    total_zips = 0
    category_details = {}
    
    for category in sorted(categories):
        category_path = os.path.join(input_path, category)
        
        # Find zip files
        zip_files = []
        other_files = []
        
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            if os.path.isfile(file_path):
                if file.endswith('.zip'):
                    zip_files.append(file)
                else:
                    other_files.append(file)
        
        category_details[category] = {
            'zip_files': zip_files,
            'other_files': other_files,
            'zip_count': len(zip_files)
        }
        
        total_zips += len(zip_files)
        
        print(f"  📁 {category:15}: {len(zip_files):3} zip files")
        if other_files:
            print(f"     └─ Also contains: {len(other_files)} other files")
    
    print(f"\n📊 Total: {total_zips} zip files across {len(categories)} categories")
    
    return category_details

def inspect_sample_zip_files(input_path, category_details, samples_per_category=2):
    """Inspect sample zip files to understand their structure."""
    
    print(f"\n🔬 Inspecting Sample ZIP Files ({samples_per_category} per category)")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp(prefix="dataset_inspect_")
    
    try:
        for category, details in category_details.items():
            if not details['zip_files']:
                continue
                
            print(f"\n📂 Category: {category}")
            print("-" * 30)
            
            # Sample first few zip files
            sample_zips = details['zip_files'][:samples_per_category]
            
            for zip_name in sample_zips:
                zip_path = os.path.join(input_path, category, zip_name)
                print(f"\n  📦 {zip_name}")
                
                try:
                    inspect_single_zip(zip_path, temp_dir)
                except Exception as e:
                    print(f"    ❌ Error inspecting {zip_name}: {e}")
                    
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def inspect_single_zip(zip_path, temp_dir):
    """Inspect a single zip file."""
    
    zip_name = os.path.basename(zip_path)
    base_name = os.path.splitext(zip_name)[0]
    
    # Extract to temporary directory
    extract_path = os.path.join(temp_dir, base_name)
    os.makedirs(extract_path, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"    📄 Contains {len(file_list)} files:")
            
            for file in file_list:
                try:
                    file_size = zip_ref.getinfo(file).file_size
                    print(f"      - {file} ({file_size:,} bytes)")
                except:
                    print(f"      - {file} (size unknown)")
            
            # Extract files
            zip_ref.extractall(extract_path)
    except zipfile.BadZipFile:
        print(f"    ❌ Bad ZIP file: {zip_name}")
        return
    except Exception as e:
        print(f"    ❌ Extraction error: {e}")
        return
    
    # Analyze extracted content
    analyze_extracted_content(extract_path, base_name)
    
    # Clean up
    shutil.rmtree(extract_path)

def analyze_extracted_content(extract_path, base_name):
    """Analyze extracted content and identify structure."""
    
    # Find all files
    all_files = []
    for root, _, files in os.walk(extract_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    
    # Categorize files
    pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
    zip_files = [f for f in all_files if f.lower().endswith('.zip')]
    other_files = [f for f in all_files if not f.lower().endswith(('.pdf', '.zip'))]
    
    print(f"    🔍 Content Analysis:")
    print(f"      📄 PDF files: {len(pdf_files)}")
    print(f"      📦 ZIP files: {len(zip_files)}")
    print(f"      📁 Other files: {len(other_files)}")
    
    # Handle different scenarios
    if len(pdf_files) == 2:
        analyze_pdf_pair(pdf_files)
    elif len(pdf_files) == 1:
        print(f"      ⚠️  Only 1 PDF found: {os.path.basename(pdf_files[0])}")
    elif len(pdf_files) == 0:
        if zip_files:
            print(f"      🔄 Contains nested ZIP files - may need recursive extraction")
            for zf in zip_files:
                print(f"        - {os.path.basename(zf)}")
        else:
            print(f"      ❌ No PDF files found")
    else:
        print(f"      ⚠️  Unexpected number of PDFs: {len(pdf_files)}")

def analyze_pdf_pair(pdf_files):
    """Analyze a pair of PDF files."""
    
    pdf_info = []
    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        try:
            pdf_size = os.path.getsize(pdf_path)
        except:
            pdf_size = 0
        
        # Determine type based on naming
        if 'info' in pdf_name.lower() or pdf_name.endswith('-info.pdf'):
            pdf_type = "📝 Metadata"
        else:
            pdf_type = "📄 Content"
        
        pdf_info.append((pdf_type, pdf_name, pdf_size))
    
    # Sort by size (metadata usually smaller)
    pdf_info.sort(key=lambda x: x[2])
    
    print(f"    🔍 PDF Analysis:")
    for pdf_type, pdf_name, pdf_size in pdf_info:
        print(f"      {pdf_type:12}: {pdf_name} ({pdf_size:,} bytes)")
    
    # Quick content peek at metadata PDF (smaller one)
    metadata_pdf_path = None
    for pdf_path in pdf_files:
        if os.path.basename(pdf_path) == pdf_info[0][1]:
            metadata_pdf_path = pdf_path
            break
    
    if metadata_pdf_path:
        peek_pdf_content(metadata_pdf_path)

def peek_pdf_content(pdf_path, max_chars=200):
    """Peek at PDF content."""
    
    try:
        # Try to import PyMuPDF
        import fitz
        
        with fitz.open(pdf_path) as doc:
            if len(doc) > 0:
                first_page = doc[0]
                text = first_page.get_text()
                
                if text.strip():
                    # Show first few lines
                    lines = text.strip().split('\n')[:5]
                    preview = '\n'.join(lines)
                    
                    if len(preview) > max_chars:
                        preview = preview[:max_chars] + "..."
                    
                    print(f"    👀 Content Preview:")
                    for line in preview.split('\n'):
                        if line.strip():
                            print(f"      | {line.strip()}")
                else:
                    print(f"    📷 Content: Image-only PDF (no text)")
            
    except ImportError:
        print(f"    ⚠️  PyMuPDF not available - install with: pip install pymupdf")
    except Exception as e:
        print(f"    ⚠️  Error reading PDF: {e}")

def check_dependencies():
    """Check if required dependencies are available."""
    
    print("\n🧪 Checking Dependencies")
    print("=" * 40)
    
    required_packages = [
        ('PyMuPDF', 'fitz', 'pip install pymupdf'),
        ('Pillow', 'PIL', 'pip install pillow'),
        ('pytesseract', 'pytesseract', 'pip install pytesseract')
    ]
    
    all_available = True
    
    for package_name, import_name, install_cmd in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name} is available")
        except ImportError:
            print(f"❌ {package_name} not found - install with: {install_cmd}")
            all_available = False
    
    return all_available

def generate_processing_recommendations(category_details):
    """Generate recommendations for processing."""
    
    print(f"\n💡 Processing Recommendations")
    print("=" * 40)
    
    total_zips = sum(d['zip_count'] for d in category_details.values())
    
    print(f"📊 Dataset Summary:")
    print(f"  Total ZIP files: {total_zips}")
    print(f"  Categories: {len(category_details)}")
    
    # Estimate processing time
    avg_time_per_zip = 10  # seconds
    total_time = total_zips * avg_time_per_zip
    
    print(f"\n⏱️  Estimated Processing Time:")
    print(f"  Sequential: ~{total_time//60} minutes")
    print(f"  Parallel (4 workers): ~{total_time//240} minutes")
    
    print(f"\n🚀 Next Steps:")
    print(f"1. Install missing dependencies (see above)")
    print(f"2. Fix the import path issues in processing scripts")
    print(f"3. Handle nested ZIP files in some categories")
    print(f"4. Test with a small subset first")
    
    print(f"\n⚠️  Issues to Address:")
    print(f"• Some ZIP files contain nested ZIPs instead of PDFs")
    print(f"• Import path needs to be fixed for processing scripts")
    print(f"• May need recursive extraction for some files")

def main():
    """Main inspection function."""
    
    print("📋 Personal RAG-LLM Dataset Inspector (Fixed Version)")
    print("=" * 60)
    
    # Get input path
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        print("Usage: python inspect_dataset_fixed.py /path/to/dataset")
        print("Example: python inspect_dataset_fixed.py /Users/yashm/Documents/GitHub/SPEC-001_Personal-RAG-LLM/SPEC-001_Personal-RAG-LLM/data/input")
        return
    
    # Step 1: Inspect dataset structure
    category_details = inspect_dataset_structure(input_path)
    
    if not category_details:
        print("\n❌ Dataset inspection failed")
        return
    
    # Step 2: Inspect sample files
    if any(details['zip_count'] > 0 for details in category_details.values()):
        inspect_sample_zip_files(input_path, category_details)
    else:
        print("\n⚠️  No zip files found to inspect")
        return
    
    # Step 3: Check dependencies
    deps_ok = check_dependencies()
    
    # Step 4: Generate recommendations
    generate_processing_recommendations(category_details)
    
    if deps_ok:
        print(f"\n✅ Ready to proceed with processing setup!")
    else:
        print(f"\n❌ Install missing dependencies first")

if __name__ == "__main__":
    main()