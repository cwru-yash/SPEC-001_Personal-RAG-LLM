# services/processor/src/pipeline/extractors/image_extractor.py
import os
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from PIL import Image
import pytesseract
import numpy as np
import io

from src.models.document import Document

class ImageExtractor:
    """Advanced image extractor with content analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize image extractor with configuration."""
        self.config = config or {}
        self.ocr_config = self.config.get("ocr", {})
        self.analysis_config = self.config.get("analysis", {})
        
    def extract(self, file_path: str) -> Document:
        """Extract and analyze image content."""
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1][1:].lower()
        
        # Create base document
        document = Document(
            doc_id=doc_id,
            file_name=file_name,
            file_extension=file_extension,
            content_type=["image", file_extension],
            created_at=datetime.now(),
            text_content="",
            metadata={}
        )
        
        try:
            # Open and analyze image
            with Image.open(file_path) as img:
                # Basic image metadata
                document.metadata.update({
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "file_size": os.path.getsize(file_path)
                })
                
                # Detect image type
                image_type = self._detect_image_type(img, file_name)
                document.metadata["image_type"] = image_type
                document.content_type.append(image_type)
                
                # Perform OCR
                ocr_text = self._extract_text_from_image(img)
                document.text_content = ocr_text
                
                # Analyze content based on type
                if image_type == "chart":
                    chart_analysis = self._analyze_chart(img, ocr_text)
                    document.metadata["chart_analysis"] = chart_analysis
                elif image_type == "diagram":
                    diagram_analysis = self._analyze_diagram(img, ocr_text)
                    document.metadata["diagram_analysis"] = diagram_analysis
                elif image_type == "floor_plan":
                    plan_analysis = self._analyze_floor_plan(img, ocr_text)
                    document.metadata["floor_plan_analysis"] = plan_analysis
                
                # Store image data for vector embedding
                document.metadata["image_data"] = self._encode_image_for_storage(img)
                
        except Exception as e:
            document.metadata["extraction_error"] = str(e)
            
        return document
    
    def _detect_image_type(self, img: Image.Image, file_name: str) -> str:
        """Detect the type of image based on content and filename."""
        file_name_lower = file_name.lower()
        
        # Check filename patterns
        if any(term in file_name_lower for term in ['chart', 'graph', 'plot']):
            return "chart"
        elif any(term in file_name_lower for term in ['diagram', 'flow', 'schema']):
            return "diagram"
        elif any(term in file_name_lower for term in ['plan', 'blueprint', 'layout']):
            return "floor_plan"
        elif any(term in file_name_lower for term in ['slide', 'presentation']):
            return "presentation_slide"
        
        # Analyze image characteristics
        # Convert to grayscale for analysis
        gray_img = img.convert('L')
        img_array = np.array(gray_img)
        
        # Check for high contrast (typical of charts/diagrams)
        std_dev = np.std(img_array)
        if std_dev > 80:  # High contrast
            # Check for grid patterns (charts)
            if self._has_grid_pattern(img_array):
                return "chart"
            else:
                return "diagram"
        
        return "general_image"
    
    def _extract_text_from_image(self, img: Image.Image) -> str:
        """Extract text from image using OCR."""
        try:
            # Preprocess image for better OCR
            processed_img = self._preprocess_for_ocr(img)
            
            # Configure tesseract
            custom_config = r'--oem 3 --psm 11'
            if self.ocr_config.get("tesseract_config"):
                custom_config = self.ocr_config["tesseract_config"]
            
            # Extract text
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            
            # Also try to extract data (for tables/structured content)
            try:
                data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
                structured_text = self._structure_ocr_data(data)
                if structured_text:
                    text += "\n\n[Structured Data]\n" + structured_text
            except:
                pass
                
            return text.strip()
            
        except Exception as e:
            return f"[OCR Error: {str(e)}]"
    
    def _preprocess_for_ocr(self, img: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        gray = img.convert('L')
        
        # Resize if too small
        min_dimension = 1000
        if gray.width < min_dimension or gray.height < min_dimension:
            ratio = min_dimension / min(gray.width, gray.height)
            new_size = (int(gray.width * ratio), int(gray.height * ratio))
            gray = gray.resize(new_size, Image.LANCZOS)
        
        # Enhance contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        return enhanced
    
    def _analyze_chart(self, img: Image.Image, ocr_text: str) -> Dict[str, Any]:
        """Analyze chart/graph images."""
        analysis = {
            "type": "chart",
            "detected_elements": []
        }
        
        # Look for chart indicators in OCR text
        text_lower = ocr_text.lower()
        
        # Detect chart type
        if any(term in text_lower for term in ['bar', 'column']):
            analysis["chart_type"] = "bar_chart"
        elif any(term in text_lower for term in ['pie', 'donut']):
            analysis["chart_type"] = "pie_chart"
        elif any(term in text_lower for term in ['line', 'trend']):
            analysis["chart_type"] = "line_chart"
        else:
            analysis["chart_type"] = "unknown"
        
        # Extract axes labels
        lines = ocr_text.split('\n')
        for line in lines:
            if any(term in line.lower() for term in ['axis', 'x:', 'y:']):
                analysis["detected_elements"].append(f"Axis label: {line.strip()}")
        
        # Extract data values (numbers)
        import re
        numbers = re.findall(r'\d+\.?\d*', ocr_text)
        if numbers:
            analysis["detected_values"] = numbers[:20]  # Limit to first 20
            
        # Extract title
        if lines:
            # First non-empty line might be title
            for line in lines:
                if line.strip() and len(line.strip()) > 10:
                    analysis["possible_title"] = line.strip()
                    break
        
        return analysis
    
    def _analyze_diagram(self, img: Image.Image, ocr_text: str) -> Dict[str, Any]:
        """Analyze diagram images."""
        analysis = {
            "type": "diagram",
            "detected_elements": []
        }
        
        # Extract key components
        components = []
        lines = ocr_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:
                components.append(line)
        
        analysis["components"] = components[:20]  # Limit to first 20
        
        # Detect diagram type based on content
        text_lower = ocr_text.lower()
        if any(term in text_lower for term in ['flow', 'process', 'step']):
            analysis["diagram_type"] = "flowchart"
        elif any(term in text_lower for term in ['architecture', 'system', 'component']):
            analysis["diagram_type"] = "architecture_diagram"
        elif any(term in text_lower for term in ['network', 'node', 'connection']):
            analysis["diagram_type"] = "network_diagram"
        else:
            analysis["diagram_type"] = "general_diagram"
        
        # Detect relationships (arrows, connections)
        if '->' in ocr_text or '→' in ocr_text:
            analysis["has_directional_flow"] = True
        
        return analysis
    
    def _analyze_floor_plan(self, img: Image.Image, ocr_text: str) -> Dict[str, Any]:
        """Analyze floor plan/blueprint images."""
        analysis = {
            "type": "floor_plan",
            "detected_rooms": [],
            "detected_measurements": []
        }
        
        # Look for room names
        room_keywords = ['room', 'bedroom', 'bathroom', 'kitchen', 'living', 'dining', 
                        'office', 'garage', 'hall', 'closet', 'patio', 'deck']
        
        lines = ocr_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            for keyword in room_keywords:
                if keyword in line_lower:
                    analysis["detected_rooms"].append(line.strip())
                    break
        
        # Extract measurements
        import re
        # Look for patterns like "12' x 14'" or "3.5m x 4.2m"
        measurements = re.findall(r'\d+\.?\d*[\'"\s]*x\s*\d+\.?\d*[\'"\s]*', ocr_text)
        analysis["detected_measurements"] = measurements[:10]  # Limit to first 10
        
        # Extract square footage if present
        sq_ft_pattern = re.findall(r'\d+\.?\d*\s*(?:sq\.?\s*ft\.?|square feet)', ocr_text.lower())
        if sq_ft_pattern:
            analysis["square_footage"] = sq_ft_pattern
        
        return analysis
    
    def _has_grid_pattern(self, img_array: np.ndarray) -> bool:
        """Detect if image has grid pattern (common in charts)."""
        # Simple edge detection to find grid lines
        from scipy import ndimage
        edges = ndimage.sobel(img_array)
        
        # Check for regular patterns in edges
        edge_density = np.sum(edges > 128) / edges.size
        return edge_density > 0.1 and edge_density < 0.4
    
    def _structure_ocr_data(self, data: Dict) -> str:
        """Structure OCR data for better understanding."""
        if not data.get('text'):
            return ""
        
        structured = []
        current_line = []
        last_top = 0
        
        for i, text in enumerate(data['text']):
            if text.strip():
                top = data['top'][i]
                # New line if vertical position changes significantly
                if abs(top - last_top) > 20 and current_line:
                    structured.append(' '.join(current_line))
                    current_line = []
                current_line.append(text)
                last_top = top
        
        if current_line:
            structured.append(' '.join(current_line))
        
        return '\n'.join(structured)
    
    def _encode_image_for_storage(self, img: Image.Image) -> Dict[str, Any]:
        """Encode image for storage and later embedding."""
        # Resize for efficient storage
        max_size = (512, 512)
        img.thumbnail(max_size, Image.LANCZOS)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr = img_byte_arr.getvalue()
        
        return {
            "size": len(img_byte_arr),
            "thumbnail_dimensions": img.size,
            "format": "JPEG"
            # Note: Actual image bytes would be stored in a blob storage
        }