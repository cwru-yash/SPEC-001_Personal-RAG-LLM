# # services/processor/src/pipeline/classifier.py
# from typing import Dict, List, Any
# import re
# from src.models.document import Document

# class DocumentClassifier:
#     """Document classification and tagging system."""
    
#     def __init__(self, config: Dict[str, Any] = None):
#         """Initialize classifier with configuration."""
#         self.config = config or {}
#         self.enabled = self.config.get("enabled", True)
    
#     def classify(self, document: Document) -> Document:
#         """Classify document and add metadata tags."""
        
#         if not self.enabled:
#             return document
        
#         # Content-based classification
#         document = self._classify_content_type(document)
        
#         # Add persuasion tags (placeholder implementation)
#         document = self._add_persuasion_tags(document)
        
#         # Add quality metrics
#         document = self._assess_quality(document)
        
#         return document
    
#     def _classify_content_type(self, document: Document) -> Document:
#         """Classify document content type based on text analysis."""
        
#         if not document.text_content:
#             return document
        
#         text_lower = document.text_content.lower()
        
#         # Academic/research content
#         if any(term in text_lower for term in ['abstract', 'methodology', 'conclusion', 'references']):
#             if 'academic' not in document.content_type:
#                 document.content_type.append('academic')
        
#         # Legal content
#         if any(term in text_lower for term in ['whereas', 'jurisdiction', 'plaintiff', 'defendant']):
#             if 'legal' not in document.content_type:
#                 document.content_type.append('legal')
        
#         # Financial content
#         if any(term in text_lower for term in ['revenue', 'profit', 'financial', 'investment']):
#             if 'financial' not in document.content_type:
#                 document.content_type.append('financial')
        
#         # Email content
#         if any(term in text_lower for term in ['subject:', 'from:', 'to:', 'dear']):
#             if 'email' not in document.content_type:
#                 document.content_type.append('email')
        
#         return document
    
#     def _add_persuasion_tags(self, document: Document) -> Document:
#         """Add basic persuasion strategy tags."""
        
#         if not document.text_content:
#             return document
        
#         persuasion_tags = []
#         text_lower = document.text_content.lower()
        
#         # Authority indicators
#         if any(term in text_lower for term in ['expert', 'authority', 'certified', 'official']):
#             persuasion_tags.append('authority')
        
#         # Social proof indicators  
#         if any(term in text_lower for term in ['popular', 'majority', 'everyone', 'most people']):
#             persuasion_tags.append('social_proof')
        
#         # Urgency indicators
#         if any(term in text_lower for term in ['urgent', 'deadline', 'limited time', 'act now']):
#             persuasion_tags.append('urgency')
        
#         document.persuasion_tags.extend(persuasion_tags)
        
#         return document
    
#     def _assess_quality(self, document: Document) -> Document:
#         """Assess document quality and add metrics."""
        
#         quality_score = 1.0
#         quality_issues = []
        
#         # Text content quality
#         if not document.text_content.strip():
#             quality_score -= 0.5
#             quality_issues.append('no_text_content')
#         elif len(document.text_content) < 100:
#             quality_score -= 0.2
#             quality_issues.append('very_short_content')
        
#         # Extraction quality
#         if 'extraction_error' in document.metadata:
#             quality_score -= 0.3
#             quality_issues.append('extraction_error')
        
#         # Store quality metrics
#         document.metadata['quality_score'] = max(0.0, quality_score)
#         if quality_issues:
#             document.metadata['quality_issues'] = quality_issues
        
#         return document

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
        
        # Add persuasion tags
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
        
        # Technical content
        if any(word in text for word in ['algorithm', 'implementation', 'architecture', 'system']):
            if 'technical' not in document.content_type:
                document.content_type.append('technical')
        
        return document
    
    def _add_persuasion_tags(self, document):
        """Add persuasion strategy tags."""
        if not hasattr(document, 'persuasion_tags') or document.persuasion_tags is None:
            document.persuasion_tags = []
        
        text = document.text_content.lower() if document.text_content else ""
        
        # Simple keyword-based persuasion detection
        if any(word in text for word in ['proven', 'expert', 'authority', 'research shows']):
            if 'authority' not in document.persuasion_tags:
                document.persuasion_tags.append('authority')
        
        if any(word in text for word in ['limited time', 'exclusive', 'rare', 'only']):
            if 'scarcity' not in document.persuasion_tags:
                document.persuasion_tags.append('scarcity')
        
        if any(word in text for word in ['everyone', 'most people', 'popular', 'trending']):
            if 'social_proof' not in document.persuasion_tags:
                document.persuasion_tags.append('social_proof')
        
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
        
        # Check for metadata completeness
        if document.author:
            quality_score += 0.1
        if document.created_at:
            quality_score += 0.1
        
        # Add quality score to metadata
        document.metadata['quality_score'] = max(0.0, min(1.0, quality_score))
        
        return document