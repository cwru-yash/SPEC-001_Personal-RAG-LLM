#!/usr/bin/env python3
"""
ZIP Dataset Processor - Fixed Version
Properly extracts ZIP files, processes PDF pairs, handles failures
Creates timestamped directory structure by file type
"""

import os
import json
import zipfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import uuid
import re

def check_dependencies():
    """Check required dependencies."""
    deps = [
        ("fitz", "pymupdf", "PDF processing"),
        ("PIL", "pillow", "Image processing"), 
        ("pytesseract", "pytesseract", "OCR")
    ]
    
    missing = []
    for module, package, desc in deps:
        try:
            __import__(module)
            print(f"✅ {desc}")
        except ImportError:
            missing.append((package, desc))
            print(f"❌ {desc} - install: pip install {package}")
    
    if missing:
        print(f"\nInstall: pip install {' '.join([p[0] for p in missing])}")
        return False
    return True

class PDFExtractor:
    """Extract text from PDF files with OCR fallback."""
    
    def extract_pdf_text(self, file_path: str) -> Tuple[str, str, List[str], bool]:
        """Extract text from PDF. Returns (text, method, errors, is_success)."""
        errors = []
        
        try:
            import fitz
            text_parts = []
            
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    # Get text
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(f"=== Page {page_num + 1} ===\n{text}")
                    
                    # OCR images if little text
                    if len(text.strip()) < 50:
                        images = page.get_images()
                        for img_index, img in enumerate(images):
                            try:
                                xref = img[0]
                                pix = fitz.Pixmap(doc, xref)
                                if pix.n - pix.alpha < 4:
                                    img_data = pix.tobytes("png")
                                    ocr_text = self._ocr_image_bytes(img_data)
                                    if ocr_text.strip():
                                        text_parts.append(f"=== Page {page_num + 1} Image {img_index + 1} (OCR) ===\n{ocr_text}")
                                pix = None
                            except Exception as e:
                                errors.append(f"Image OCR error: {str(e)}")
            
            full_text = "\n\n".join(text_parts)
            
            # Check if extraction was successful
            success = self._is_valid_text(full_text)
            
            return full_text, "pymupdf+ocr", errors, success
            
        except ImportError:
            errors.append("PyMuPDF not available")
            return "", "no_pdf_library", errors, False
        except Exception as e:
            errors.append(f"PDF error: {str(e)}")
            return "", "error", errors, False
    
    def _ocr_image_bytes(self, img_bytes: bytes) -> str:
        """OCR from image bytes."""
        try:
            from PIL import Image
            import pytesseract
            import io
            
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img, config=r'--oem 3 --psm 11')
            return text.strip()
        except:
            return ""
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if extracted text is valid (not gibberish/binary)."""
        if not text or len(text.strip()) < 10:
            return False
        
        # Check for binary/gibberish patterns
        binary_patterns = [
            r'PK\x03\x04',  # ZIP header
            r'\\u[0-9a-fA-F]{4}',  # Unicode escape sequences
            r'[^\x20-\x7E\s]{20,}',  # Long sequences of non-printable chars
        ]
        
        for pattern in binary_patterns:
            if re.search(pattern, text):
                return False
        
        # Check ratio of printable characters
        printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
        if len(text) > 0 and printable_chars / len(text) < 0.7:
            return False
        
        return True

class ZipDatasetProcessor:
    """Process ZIP dataset with PDF pairs."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamped_dir = self.output_dir / timestamp
        self.timestamped_dir.mkdir(exist_ok=True)
        
        # Create base directories
        self.txt_base = self.timestamped_dir / "txt_files"
        self.json_base = self.timestamped_dir / "json_files" 
        
        self.txt_base.mkdir(exist_ok=True)
        self.json_base.mkdir(exist_ok=True)
        
        self.pdf_extractor = PDFExtractor()
        self.stats = {
            "processed_zips": 0,
            "failed_zips": 0,
            "document_pairs": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "by_category": {}
        }
        
        print(f"📁 Output directory: {self.timestamped_dir}")
    
    def process_directory(self, input_dir: str) -> None:
        """Process all ZIP files in directory structure."""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return
        
        # Find all ZIP files by category
        zip_files = []
        for root, dirs, files in os.walk(input_path):
            category = Path(root).name
            if category != Path(input_dir).name:  # Skip root directory name
                for file in files:
                    if file.endswith('.zip'):
                        zip_path = Path(root) / file
                        zip_files.append((zip_path, category))
        
        print(f"📁 Found {len(zip_files)} ZIP files to process")
        
        # Process each ZIP
        for zip_path, category in zip_files:
            print(f"🔄 Processing: {category}/{zip_path.name}")
            self._process_zip_file(zip_path, category)
        
        # Save summary
        self._save_summary()
        self._print_summary()
    
    def _process_zip_file(self, zip_path: Path, category: str) -> None:
        """Process a single ZIP file."""
        try:
            # Create category directories
            category_txt_dir = self.txt_base / category
            category_json_dir = self.json_base / category
            category_txt_dir.mkdir(exist_ok=True)
            category_json_dir.mkdir(exist_ok=True)
            
            # Create failed directories
            failed_txt_dir = category_txt_dir / "failed_files"
            failed_json_dir = category_json_dir / "failed_files"
            failed_txt_dir.mkdir(exist_ok=True)
            failed_json_dir.mkdir(exist_ok=True)
            
            # Create temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract ZIP
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)
                
                # Find PDF files
                pdf_files = list(temp_path.glob("*.pdf"))
                
                if not pdf_files:
                    print(f"⚠️  No PDFs in {zip_path.name}")
                    return
                
                # Separate content and info PDFs
                content_pdfs = []
                info_pdfs = []
                
                for pdf in pdf_files:
                    if "-info" in pdf.stem:
                        info_pdfs.append(pdf)
                    else:
                        content_pdfs.append(pdf)
                
                # Process pairs
                for content_pdf in content_pdfs:
                    base_name = content_pdf.stem
                    info_pdf = None
                    
                    # Find matching info PDF
                    for info in info_pdfs:
                        if info.stem.replace("-info", "") == base_name:
                            info_pdf = info
                            break
                    
                    # Process the pair
                    success = self._process_pdf_pair(
                        content_pdf, info_pdf, category, zip_path.stem,
                        category_txt_dir, category_json_dir,
                        failed_txt_dir, failed_json_dir
                    )
                    
                    self.stats["document_pairs"] += 1
                    if success:
                        self.stats["successful_extractions"] += 1
                    else:
                        self.stats["failed_extractions"] += 1
            
            self.stats["processed_zips"] += 1
            self.stats["by_category"][category] = self.stats["by_category"].get(category, 0) + 1
            
        except Exception as e:
            print(f"❌ Failed to process {zip_path.name}: {str(e)}")
            self.stats["failed_zips"] += 1
    
    def _process_pdf_pair(self, content_pdf: Path, info_pdf: Path, category: str, 
                         zip_name: str, txt_dir: Path, json_dir: Path,
                         failed_txt_dir: Path, failed_json_dir: Path) -> bool:
        """Process a content PDF and its metadata PDF."""
        doc_id = str(uuid.uuid4())[:8]
        base_name = content_pdf.stem
        
        # Extract content PDF
        content_text, content_method, content_errors, content_success = self.pdf_extractor.extract_pdf_text(str(content_pdf))
        
        # Extract info PDF if exists
        info_text = ""
        info_method = "none"
        info_errors = []
        info_success = False
        
        if info_pdf and info_pdf.exists():
            info_text, info_method, info_errors, info_success = self.pdf_extractor.extract_pdf_text(str(info_pdf))
        
        # Create result structure
        result = {
            "document_id": doc_id,
            "zip_file": zip_name + ".zip",
            "category": category,
            "content_file": {
                "file_name": content_pdf.name,
                "file_path": f"{category}/{zip_name}.zip/{content_pdf.name}",
                "extracted_text": content_text,
                "extraction_method": content_method,
                "errors": content_errors,
                "success": content_success
            },
            "metadata_file": {
                "file_name": info_pdf.name if info_pdf else "not_found",
                "file_path": f"{category}/{zip_name}.zip/{info_pdf.name}" if info_pdf else "not_found",
                "extracted_text": info_text,
                "extraction_method": info_method,
                "errors": info_errors,
                "success": info_success
            },
            "processed_at": datetime.now().isoformat(),
            "overall_success": content_success
        }
        
        # Determine if this is a failure
        is_failure = not content_success
        
        # Choose output directories
        output_txt_dir = failed_txt_dir if is_failure else txt_dir
        output_json_dir = failed_json_dir if is_failure else json_dir
        
        # Save outputs
        self._save_outputs(result, base_name, doc_id, output_txt_dir, output_json_dir, is_failure)
        
        return content_success
    
    def _save_outputs(self, result: Dict[str, Any], base_name: str, doc_id: str,
                     txt_dir: Path, json_dir: Path, is_failure: bool) -> None:
        """Save result as TXT and JSON files."""
        
        status = "FAILED" if is_failure else "SUCCESS"
        
        # Save content TXT
        content_txt_file = txt_dir / f"{base_name}_content_{doc_id}.txt"
        with open(content_txt_file, 'w', encoding='utf-8') as f:
            f.write(f"STATUS: {status}\n")
            f.write(f"Document: {result['content_file']['file_path']}\n")
            f.write(f"Category: {result['category']}\n")
            f.write(f"Processed: {result['processed_at']}\n")
            f.write(f"Method: {result['content_file']['extraction_method']}\n")
            if result['content_file']['errors']:
                f.write(f"Errors: {', '.join(result['content_file']['errors'])}\n")
            f.write("=" * 50 + "\n\n")
            f.write(result['content_file']['extracted_text'])
        
        # Save metadata TXT if available
        if result['metadata_file']['success']:
            metadata_txt_file = txt_dir / f"{base_name}_metadata_{doc_id}.txt"
            with open(metadata_txt_file, 'w', encoding='utf-8') as f:
                f.write(f"Metadata for: {result['content_file']['file_path']}\n")
                f.write(f"Category: {result['category']}\n")
                f.write("=" * 50 + "\n\n")
                f.write(result['metadata_file']['extracted_text'])
        
        # Save complete JSON
        json_file = json_dir / f"{base_name}_{doc_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        status_emoji = "❌" if is_failure else "✅"
        print(f"{status_emoji} {status}: {base_name}")
    
    def _save_summary(self) -> None:
        """Save processing summary."""
        summary = {
            "processing_summary": self.stats,
            "processed_at": datetime.now().isoformat(),
            "output_directory": str(self.timestamped_dir),
            "directory_structure": {
                "txt_files": str(self.txt_base),
                "json_files": str(self.json_base),
                "failed_files": "within each category directory"
            }
        }
        
        with open(self.timestamped_dir / "processing_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _print_summary(self) -> None:
        """Print processing summary."""
        print(f"\n📊 Processing Summary:")
        print(f"   ZIP files processed: {self.stats['processed_zips']}")
        print(f"   ZIP files failed: {self.stats['failed_zips']}")
        print(f"   Document pairs: {self.stats['document_pairs']}")
        print(f"   Successful extractions: {self.stats['successful_extractions']}")
        print(f"   Failed extractions: {self.stats['failed_extractions']}")
        print(f"   Success rate: {self.stats['successful_extractions']/(self.stats['document_pairs'] or 1)*100:.1f}%")
        print(f"   By category:")
        for category, count in self.stats['by_category'].items():
            print(f"     {category}: {count}")
        print(f"   Output: {self.timestamped_dir}")

def main():
    """Main function."""
    import sys
    
    print("🚀 ZIP Dataset Processor v2.0")
    print("=" * 35)
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Install missing dependencies first")
        return
    
    # Get paths
    if len(sys.argv) < 3:
        print("\nUsage: python zip_processor.py INPUT_DIR OUTPUT_DIR")
        print("\nExample:")
        print("python zip_processor.py /path/to/data/input /path/to/outputs")
        return
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Process dataset
    processor = ZipDatasetProcessor(output_dir)
    processor.process_directory(input_dir)
    
    print(f"\n✅ Processing complete!")

if __name__ == "__main__":
    main()