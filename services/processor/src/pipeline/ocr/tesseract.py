# src/pipeline/ocr/tesseract.py
import tempfile
import os
from PIL import Image
from io import BytesIO
import pytesseract
from typing import Dict, Any, Optional

class TesseractOCR:
    """OCR implementation using Tesseract."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration."""
        self.config = config or {}
        
        # Set tesseract command if provided
        if "tesseract_cmd" in self.config:
            pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_cmd"]
    
    def extract_text(self, image_data: Dict) -> str:
        """Extract text from image data."""
        try:
            # Get image bytes and save to temp file
            image_bytes = image_data.get("image")
            
            if not image_bytes:
                return ""
                
            # Create a PIL Image from bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_filename = temp_file.name
                
                # Write image bytes to temp file
                temp_file.write(image_bytes)
                
            # Process with OCR
            with Image.open(temp_filename) as img:
                # Extract text
                text = pytesseract.image_to_string(
                    img, 
                    lang=self.config.get("lang", "eng"),
                    config=self.config.get("tesseract_config", "")
                )
                
            # Clean up
            os.unlink(temp_filename)
            
            return text.strip()
            
        except Exception as e:
            print(f"OCR error: {e}")
            return ""