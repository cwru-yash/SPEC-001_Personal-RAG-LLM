# src/models/document.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any

@dataclass
class Document:
    """Represents a processed document in the system."""
    doc_id: str
    file_name: str
    file_extension: str
    content_type: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    author: Optional[str] = None
    text_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    persuasion_tags: List[str] = field(default_factory=list)
    box_folder: Optional[str] = None
    
    # Will be populated after chunking and embedding
    chunks: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    chunk_id: str
    doc_id: str
    text_chunk: str
    embedding: Optional[List[float]] = None
    tag_context: List[str] = field(default_factory=list)