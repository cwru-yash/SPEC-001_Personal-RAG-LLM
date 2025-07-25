# services/processor/src/pipeline/enhanced_ocr_processor.py

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import fitz
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class OCRResult:
    """Result from enhanced OCR processing."""
    success: bool
    extracted_text: str
    confidence: float
    processing_method: str
    preprocessing_applied: List[str]
    error_message: Optional[str] = None

class EnhancedOCRProcessor:
    """Enhanced OCR processor with preprocessing and multiple engine support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('vlm', {}).get('fallback', {}).get('enhanced_ocr', {})
        self.preprocessing_enabled = self.config.get('preprocessing', {})
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # OCR configuration
        self.tesseract_config = self.config.get('tesseract_config', '--psm 3')
        
        self.logger.info("Enhanced OCR processor initialized")
    
    async def process_document_with_progress(self, pdf_path: str) -> Dict[str, Any]:
        """Process document using enhanced OCR with preprocessing."""
        
        try:
            doc = fitz.open(pdf_path)
            page_results = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                page_image = self._page_to_image(page)
                
                # Apply preprocessing
                enhanced_image, preprocessing_steps = self._preprocess_image(page_image)
                
                # Extract text using OCR
                ocr_result = self._extract_text_with_ocr(enhanced_image, preprocessing_steps)
                
                page_results.append({
                    'page_number': page_num,
                    'ocr_result': ocr_result,
                    'preprocessing_applied': preprocessing_steps
                })
            
            doc.close()
            
            # Aggregate results
            combined_text = '\n\n'.join([
                f"--- Page {r['page_number'] + 1} ---\n{r['ocr_result'].extracted_text}"
                for r in page_results if r['ocr_result'].success
            ])
            
            return {
                'processing_method': 'enhanced_ocr_fallback',
                'pages': page_results,
                'document_summary': {
                    'combined_text_content': combined_text,
                    'total_pages': len(page_results),
                    'successful_pages': len([r for r in page_results if r['ocr_result'].success])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced OCR processing failed: {str(e)}")
            raise
    
    def _page_to_image(self, page: fitz.Page) -> Image.Image:
        """Convert PDF page to high-quality image for OCR."""
        # Use high resolution for better OCR results
        matrix = fitz.Matrix(3.0, 3.0)  # 3x scaling for crisp text
        pix = page.get_pixmap(matrix=matrix)
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
    
    def _preprocess_image(self, image: Image.Image) -> tuple[Image.Image, List[str]]:
        """Apply preprocessing steps to improve OCR accuracy."""
        
        preprocessing_steps = []
        enhanced_image = image.copy()
        
        # Convert to numpy array for OpenCV operations
        img_array = np.array(enhanced_image)
        
        # Apply contrast enhancement if enabled
        if self.preprocessing_enabled.get('enhance_contrast', False):
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(1.3)
            preprocessing_steps.append('contrast_enhancement')
        
        # Apply noise reduction if enabled
        if self.preprocessing_enabled.get('noise_reduction', False):
            # Convert back to numpy for OpenCV noise reduction
            img_array = np.array(enhanced_image)
            img_array = cv2.fastNlMeansDenoising(img_array)
            enhanced_image = Image.fromarray(img_array)
            preprocessing_steps.append('noise_reduction')
        
        # Apply deskewing if enabled
        if self.preprocessing_enabled.get('deskew', False):
            enhanced_image = self._deskew_image(enhanced_image)
            preprocessing_steps.append('deskewing')
        
        return enhanced_image, preprocessing_steps
    
    def _deskew_image(self, image: Image.Image) -> Image.Image:
        """Apply deskewing to correct rotated text."""
        try:
            # Convert to grayscale and then to numpy array
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Find text lines using HoughLines
            edges = cv2.Canny(img_array, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate average angle
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    if angle < 45:
                        angles.append(angle)
                    elif angle > 135:
                        angles.append(angle - 180)
                
                if angles:
                    avg_angle = np.mean(angles)
                    # Rotate image to correct skew
                    rotated = image.rotate(avg_angle, expand=True, fillcolor='white')
                    return rotated
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Deskewing failed: {e}")
            return image
    
    def _extract_text_with_ocr(self, image: Image.Image, preprocessing_steps: List[str]) -> OCRResult:
        """Extract text using Tesseract OCR with confidence scoring."""
        
        try:
            # Extract text with confidence data
            ocr_data = pytesseract.image_to_data(
                image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Extract text
            extracted_text = pytesseract.image_to_string(image, config=self.tesseract_config)
            
            return OCRResult(
                success=True,
                extracted_text=extracted_text.strip(),
                confidence=avg_confidence / 100.0,  # Convert to 0-1 scale
                processing_method='tesseract_enhanced',
                preprocessing_applied=preprocessing_steps
            )
            
        except Exception as e:
            return OCRResult(
                success=False,
                extracted_text='',
                confidence=0.0,
                processing_method='tesseract_enhanced',
                preprocessing_applied=preprocessing_steps,
                error_message=str(e)
            )