#!/usr/bin/env python3
"""
Fixed ZIP Dataset Processor
Extracts ZIP contents and processes PDF pairs with timestamp-based output structure
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
    """Extract text from PDF files with enhanced OCR."""
    
    def extract_pdf_text(self, file_path: str) -> Tuple[str, str, List[str]]:
        """Extract text from PDF with aggressive OCR for images."""
        errors = []
        
        try:
            import fitz
            text_parts = []
            
            with fitz.open(file_path) as doc:
                total_text_length = 0
                
                for page_num, page in enumerate(doc):
                    # Get existing text
                    text = page.get_text().strip()
                    total_text_length += len(text)
                    
                    if text:
                        text_parts.append(f"=== Page {page_num + 1} ===\n{text}")
                    
                    # Always try OCR for images, especially if little text
                    images = page.get_images()
                    if images:
                        for img_index, img in enumerate(images):
                            try:
                                xref = img[0]
                                pix = fitz.Pixmap(doc, xref)
                                if pix.n - pix.alpha < 4:  # RGB or GRAY
                                    img_data = pix.tobytes("png")
                                    ocr_text = self._ocr_image_bytes(img_data)
                                    if ocr_text.strip():
                                        text_parts.append(f"=== Page {page_num + 1} Image {img_index + 1} (OCR) ===\n{ocr_text}")
                                pix = None
                            except Exception as e:
                                errors.append(f"Image {img_index} OCR error: {str(e)}")
                
                # If very little text found, try full page OCR
                if total_text_length < 100:
                    for page_num, page in enumerate(doc):
                        try:
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
                            img_data = pix.tobytes("png")
                            page_ocr = self._ocr_image_bytes(img_data)
                            if page_ocr.strip():
                                text_parts.append(f"=== Page {page_num + 1} (Full Page OCR) ===\n{page_ocr}")
                            pix = None
                        except Exception as e:
                            errors.append(f"Page {page_num + 1} full OCR error: {str(e)}")
            
            final_text = "\n\n".join(text_parts)
            method = "pymupdf+enhanced_ocr" if text_parts else "pymupdf_no_content"
            
            return final_text, method, errors
            
        except ImportError:
            errors.append("PyMuPDF not available")
            return "", "no_pdf_library", errors
        except Exception as e:
            errors.append(f"PDF error: {str(e)}")
            return "", "error", errors
    
    def _ocr_image_bytes(self, img_bytes: bytes) -> str:
        """Enhanced OCR from image bytes."""
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import pytesseract
            import io
            
            img = Image.open(io.BytesIO(img_bytes))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Image preprocessing for better OCR
            img = img.filter(ImageFilter.MedianFilter())  # Remove noise
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)  # Increase contrast
            
            # Try multiple OCR configurations
            configs = [
                r'--oem 3 --psm 6',   # Single block
                r'--oem 3 --psm 11',  # Sparse text  
                r'--oem 3 --psm 12',  # Single block, sparse
                r'--oem 3 --psm 4'    # Single column
            ]
            
            best_text = ""
            for config in configs:
                try:
                    text = pytesseract.image_to_string(img, config=config)
                    if len(text.strip()) > len(best_text.strip()):
                        best_text = text
                except:
                    continue
            
            return best_text.strip()
            
        except Exception as e:
            return f"[OCR Error: {str(e)}]"

class ZipDatasetProcessor:
    """Process ZIP dataset with timestamp-organized output."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp-based subdirectory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = self.output_dir / timestamp
        
        # Create output structure
        self.txt_base = self.session_dir / "txt_files"
        self.json_base = self.session_dir / "json_files"
        self.metadata_base = self.session_dir / "metadata"
        
        for base_dir in [self.txt_base, self.json_base, self.metadata_base]:
            base_dir.mkdir(parents=True, exist_ok=True)
        
        self.pdf_extractor = PDFExtractor()
        self.stats = {
            "processed_zips": 0,
            "failed_zips": 0,
            "document_pairs": 0,
            "by_category": {},
            "session_timestamp": timestamp
        }
        
        print(f"📁 Session output: {self.session_dir}")
    
    def process_directory(self, input_dir: str) -> None:
        """Process all ZIP files in directory structure."""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return
        
        # Find all ZIP files with their categories
        zip_files = []
        for root, dirs, files in os.walk(input_path):
            category = Path(root).relative_to(input_path).parts[0] if Path(root) != input_path else "Uncategorized"
            
            for file in files:
                if file.endswith('.zip'):
                    zip_path = Path(root) / file
                    zip_files.append((zip_path, category))
        
        print(f"📁 Found {len(zip_files)} ZIP files across categories")
        
        # Create category directories
        categories = set(cat for _, cat in zip_files)
        for category in categories:
            (self.txt_base / category).mkdir(exist_ok=True)
            (self.json_base / category).mkdir(exist_ok=True)
            (self.metadata_base / category).mkdir(exist_ok=True)
        
        # Process each ZIP
        for zip_path, category in zip_files:
            print(f"🔄 Processing: {category}/{zip_path.name}")
            self._process_zip_file(zip_path, category)
        
        # Save summary
        self._save_summary()
        self._print_summary()
    
    def _process_zip_file(self, zip_path: Path, category: str) -> None:
        """Process a single ZIP file by extracting contents."""
        try:
            # Create temp directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract ZIP contents
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)
                    print(f"   📦 Extracted to temp: {list(temp_path.glob('*'))}")
                
                # Find PDF files in extracted content
                pdf_files = list(temp_path.glob("*.pdf"))
                
                if not pdf_files:
                    print(f"   ⚠️  No PDFs found in {zip_path.name}")
                    return
                
                # Separate content and info PDFs
                content_pdfs = []
                info_pdfs = []
                
                for pdf in pdf_files:
                    if "-info" in pdf.stem.lower():
                        info_pdfs.append(pdf)
                    else:
                        content_pdfs.append(pdf)
                
                print(f"   📄 Found {len(content_pdfs)} content PDFs, {len(info_pdfs)} info PDFs")
                
                # Process pairs
                for content_pdf in content_pdfs:
                    base_name = content_pdf.stem
                    info_pdf = None
                    
                    # Find matching info PDF
                    for info in info_pdfs:
                        if info.stem.lower().replace("-info", "") == base_name.lower():
                            info_pdf = info
                            break
                    
                    # Process the pair
                    self._process_pdf_pair(content_pdf, info_pdf, category, zip_path.stem)
                    self.stats["document_pairs"] += 1
            
            self.stats["processed_zips"] += 1
            self.stats["by_category"][category] = self.stats["by_category"].get(category, 0) + 1
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            self.stats["failed_zips"] += 1
    
    def _process_pdf_pair(self, content_pdf: Path, info_pdf: Path, category: str, zip_name: str) -> None:
        """Process content PDF and metadata PDF pair."""
        doc_id = str(uuid.uuid4())[:8]
        base_name = content_pdf.stem
        
        print(f"     🔍 Processing: {base_name}")
        
        # Extract content PDF
        content_text, content_method, content_errors = self.pdf_extractor.extract_pdf_text(str(content_pdf))
        
        # Extract info PDF if exists
        info_text = ""
        info_method = "none"
        info_errors = []
        
        if info_pdf and info_pdf.exists():
            print(f"     📋 Processing metadata: {info_pdf.name}")
            info_text, info_method, info_errors = self.pdf_extractor.extract_pdf_text(str(info_pdf))
        
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
                "text_length": len(content_text),
                "errors": content_errors,
                "success": len(content_text.strip()) > 0
            },
            "metadata_file": {
                "file_name": info_pdf.name if info_pdf else "not_found",
                "file_path": f"{category}/{zip_name}.zip/{info_pdf.name}" if info_pdf else "not_found",
                "extracted_text": info_text,
                "extraction_method": info_method,
                "text_length": len(info_text),
                "errors": info_errors,
                "success": len(info_text.strip()) > 0
            },
            "processed_at": datetime.now().isoformat()
        }
        
        # Save outputs in category-organized structure
        self._save_outputs(result, base_name, doc_id, category)
        
        print(f"     ✅ Saved: {content_text[:50] if content_text else 'No text'}...")
    
    def _save_outputs(self, result: Dict[str, Any], base_name: str, doc_id: str, category: str) -> None:
        """Save result with timestamp and category organization."""
        
        # Save content TXT
        txt_dir = self.txt_base / category
        content_txt_file = txt_dir / f"{base_name}_content_{doc_id}.txt"
        
        with open(content_txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Document: {result['content_file']['file_path']}\n")
            f.write(f"Category: {result['category']}\n")
            f.write(f"Processed: {result['processed_at']}\n")
            f.write(f"Method: {result['content_file']['extraction_method']}\n")
            f.write(f"Text Length: {result['content_file']['text_length']} chars\n")
            if result['content_file']['errors']:
                f.write(f"Errors: {'; '.join(result['content_file']['errors'])}\n")
            f.write("=" * 50 + "\n\n")
            f.write(result['content_file']['extracted_text'])
        
        # Save metadata TXT if available
        if result['metadata_file']['success']:
            metadata_dir = self.metadata_base / category
            metadata_txt_file = metadata_dir / f"{base_name}_metadata_{doc_id}.txt"
            
            with open(metadata_txt_file, 'w', encoding='utf-8') as f:
                f.write(f"Metadata for: {result['content_file']['file_path']}\n")
                f.write(f"Category: {result['category']}\n")
                f.write("=" * 50 + "\n\n")
                f.write(result['metadata_file']['extracted_text'])
        
        # Save complete JSON
        json_dir = self.json_base / category
        json_file = json_dir / f"{base_name}_{doc_id}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    def _save_summary(self) -> None:
        """Save processing summary."""
        summary = {
            "processing_summary": self.stats,
            "processed_at": datetime.now().isoformat(),
            "session_directory": str(self.session_dir),
            "output_structure": {
                "txt_files": str(self.txt_base),
                "json_files": str(self.json_base),
                "metadata": str(self.metadata_base)
            }
        }
        
        with open(self.session_dir / "processing_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _print_summary(self) -> None:
        """Print processing summary."""
        print(f"\n📊 Processing Summary ({self.stats['session_timestamp']}):")
        print(f"   ZIP files processed: {self.stats['processed_zips']}")
        print(f"   ZIP files failed: {self.stats['failed_zips']}")
        print(f"   Document pairs: {self.stats['document_pairs']}")
        print(f"   Categories processed:")
        for category, count in self.stats['by_category'].items():
            print(f"     {category}: {count} ZIP files")
        print(f"   Session output: {self.session_dir}")

def main():
    """Main function."""
    import sys
    
    print("🚀 Fixed ZIP Dataset Processor")
    print("=" * 35)
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Install missing dependencies first")
        return
    
    # Get paths
    if len(sys.argv) < 3:
        print("\nUsage: python zip_processor_fixed.py INPUT_DIR OUTPUT_DIR")
        print("\nExample:")
        print("python zip_processor_fixed.py /path/to/data/input /path/to/outputs")
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