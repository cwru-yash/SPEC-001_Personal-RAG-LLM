# src/pipeline/ocr/element_detector.py
from typing import List, Dict, Any

class VisualElementDetector:
    """Detects visual elements like images and charts in PDF pages."""
    
    def detect_images(self, page) -> List[Any]:
        """Detect images in a PDF page."""
        images = []
        
        # Extract images using PyMuPDF
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]  # Image reference
            
            # Get image info
            base_image = page.parent.extract_image(xref)
            if base_image:
                images.append({
                    "xref": xref,
                    "page_num": page.number,
                    "image": base_image
                })
        
        return images
    
    def detect_charts(self, page) -> List[Any]:
        """Detect charts and diagrams in a PDF page."""
        # Detecting charts is more complex and might require ML models
        # For a simple approach, we'll look for image blocks with specific characteristics
        
        charts = []
        
        # Get all drawing blocks (vector graphics)
        drawings = []
        for block in page.get_drawings():
            if block:
                drawings.append(block)
        
        # Simple heuristic: areas with many vector graphics might be charts
        # Group nearby drawings as potential charts
        if len(drawings) > 5:  # Arbitrary threshold
            # For simplicity, we'll consider the entire drawing area as one chart
            # In a real implementation, clustering would be better
            charts.append({
                "page_num": page.number,
                "drawings": drawings,
                "rect": page.rect  # Use page rect as approximation
            })
        
        return charts