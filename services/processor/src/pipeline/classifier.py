# services/processor/src/pipeline/classifier.py
from typing import Dict, Any, List
import re
from datetime import datetime

class DocumentClassifier:
    """Document classification and tagging system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize classifier with configuration."""
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
    
    def classify(self, document) -> Any:
        """Classify document and add tags."""
        if not self.enabled:
            return document
        
        # Add content classification
        document = self._classify_content_type(document)
        
        # Add persuasion tags (placeholder)
        document = self._add_persuasion_tags(document)
        
        # Add quality metrics
        document = self._assess_quality(document)
        
        return document
    
    def _classify_content_type(self, document):
        """Classify document content type."""
        text = document.text_content.lower() if document.text_content else ""
        
        # Academic/research content
        if any(word in text for word in ['abstract', 'methodology', 'conclusion', 'references']):
            if 'academic' not in document.content_type:
                document.content_type.append('academic')
        
        # Legal content
        if any(word in text for word in ['agreement', 'contract', 'legal', 'court']):
            if 'legal' not in document.content_type:
                document.content_type.append('legal')
        
        # Financial content
        if any(word in text for word in ['financial', 'revenue', 'profit', 'investment']):
            if 'financial' not in document.content_type:
                document.content_type.append('financial')
        
        return document
    
    def _add_persuasion_tags(self, document):
        """Add persuasion strategy tags (placeholder)."""
        if not hasattr(document, 'persuasion_tags'):
            document.persuasion_tags = []
        
        text = document.text_content.lower() if document.text_content else ""
        
        # Simple keyword-based persuasion detection
        if any(word in text for word in ['proven', 'expert', 'authority']):
            document.persuasion_tags.append('authority')
        
        if any(word in text for word in ['limited time', 'exclusive', 'rare']):
            document.persuasion_tags.append('scarcity')
        
        return document
    
    def _assess_quality(self, document):
        """Assess document quality."""
        quality_score = 1.0
        
        # Text length check
        if document.text_content:
            text_length = len(document.text_content)
            if text_length < 100:
                quality_score -= 0.3
            elif text_length > 1000:
                quality_score += 0.2
        else:
            quality_score -= 0.5
        
        # Add quality score to metadata
        document.metadata['quality_score'] = max(0.0, min(1.0, quality_score))
        
        return document