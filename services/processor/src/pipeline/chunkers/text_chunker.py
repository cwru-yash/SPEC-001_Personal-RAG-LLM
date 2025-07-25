# src/pipeline/chunkers/text_chunker.py
import uuid
import re
from typing import List, Dict, Any
from src.models.document import Document, DocumentChunk

class ContentAwareChunker:
    """Chunks document text based on content type."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration."""
        self.config = config or {}
        
        # Default chunking parameters
        self.default_chunk_size = self.config.get("default_chunk_size", 500)
        self.default_chunk_overlap = self.config.get("default_chunk_overlap", 50)
        
        # Content-specific parameters
        self.email_chunk_size = self.config.get("email_chunk_size", 300)
        self.email_chunk_overlap = self.config.get("email_chunk_overlap", 30)
        
        self.presentation_chunk_size = self.config.get("presentation_chunk_size", 250)
        self.presentation_chunk_overlap = self.config.get("presentation_chunk_overlap", 25)
    
    # def chunk_document(self, document: Document) -> Document:
    #     """Chunk document based on its content type."""
    #     # Determine chunking strategy based on content type
    #     if "email" in document.content_type:
    #         document = self._chunk_email(document)
    #     elif "presentation" in document.content_type:
    #         document = self._chunk_presentation(document)
    #     else:
    #         document = self._chunk_regular_document(document)
            
    #     return document

    def chunk_document(self, document: Document) -> Document:
        """Chunk document based on its content type."""
        # Determine chunking strategy based on content type
        if "email" in document.content_type:
            document = self._chunk_email(document)
        elif "presentation" in document.content_type:
            document = self._chunk_presentation(document)
        elif "image" in document.content_type:
            document = self._chunk_image(document)
        else:
            document = self._chunk_regular_document(document)
            
        return document

    def _chunk_image(self, document: Document) -> Document:
        """Chunk an image document with content awareness."""
        text = document.text_content
        chunks = []
        
        if not text.strip():
            # Empty text - create empty chunk
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=document.doc_id,
                text_chunk="",
                tag_context=document.persuasion_tags + ["image"]
            )
            chunks.append(chunk.__dict__)
            document.chunks = chunks
            return document
        
        # For images, we might have OCR text from different visual elements
        # Split by visual element markers if present
        if "[Visual Element" in text:
            elements = re.split(r'\[Visual Element[^\]]*\]:', text)
            
            # First element might be general text
            if elements[0].strip():
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=document.doc_id,
                    text_chunk=elements[0].strip(),
                    tag_context=document.persuasion_tags + ["image", "general"]
                )
                chunks.append(chunk.__dict__)
            
            # Process each visual element
            for i, element in enumerate(elements[1:], 1):
                if element.strip():
                    chunk = DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=document.doc_id,
                        text_chunk=element.strip(),
                        tag_context=document.persuasion_tags + ["image", f"visual_element_{i}"]
                    )
                    chunks.append(chunk.__dict__)
        else:
            # No visual elements marked - create single chunk
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=document.doc_id,
                text_chunk=text.strip(),
                tag_context=document.persuasion_tags + ["image"]
            )
            chunks.append(chunk.__dict__)
        
        document.chunks = chunks
        return document
    
    def _chunk_regular_document(self, document: Document) -> Document:
        """Chunk a regular document with paragraph awareness."""
        text = document.text_content
        chunks = []
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, create a new chunk
            if len(current_chunk) + len(paragraph) > self.default_chunk_size:
                if current_chunk:
                    chunk = DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=document.doc_id,
                        text_chunk=current_chunk.strip(),
                        tag_context=document.persuasion_tags
                    )
                    chunks.append(chunk.__dict__)
                
                # Start new chunk with overlap from end of previous chunk
                if current_chunk and self.default_chunk_overlap > 0:
                    # Add overlap from previous chunk
                    words = current_chunk.split()
                    overlap_word_count = min(len(words), self.default_chunk_overlap)
                    overlap_text = ' '.join(words[-overlap_word_count:])
                    current_chunk = overlap_text + ' ' + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk if not empty
        if current_chunk:
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=document.doc_id,
                text_chunk=current_chunk.strip(),
                tag_context=document.persuasion_tags
            )
            chunks.append(chunk.__dict__)
            
        # Update document with chunks
        document.chunks = chunks
        return document
    
    def _chunk_email(self, document: Document) -> Document:
        """Chunk an email with header and thread awareness."""
        # For emails, extract metadata and keep each email in thread as separate chunk
        
        # Get full text
        text = document.text_content
        chunks = []
        
        # Split into email thread parts if available
        # Use thread markers to detect parts
        thread_markers = [
            r'----+ ?Original Message ?----+',
            r'----+ ?Forwarded Message ?----+',
            r'On .* wrote:',
            r'From:.*Sent:.*To:.*Subject:'
        ]
        
        # Combine markers into single regex
        combined_regex = '|'.join(thread_markers)
        
        # Split by thread markers
        thread_parts = re.split(combined_regex, text)
        
        # Create chunk for each thread part
        for i, part in enumerate(thread_parts):
            # Skip empty parts
            if not part.strip():
                continue
                
            # Create chunk
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=document.doc_id,
                text_chunk=part.strip(),
                tag_context=document.persuasion_tags + [f"email_part_{i}"]
            )
            chunks.append(chunk.__dict__)
            
        # If we couldn't split into thread parts, chunk by size
        if not chunks:
            # Fall back to regular chunking
            return self._chunk_regular_document(document)
            
        # Update document with chunks
        document.chunks = chunks
        return document
    
    def _chunk_presentation(self, document: Document) -> Document:
        """Chunk a presentation with slide awareness."""
        text = document.text_content
        chunks = []
        
        # Split by slide markers
        slide_markers = re.finditer(r'---\s*SLIDE\s+(\d+)\s*---', text)
        slide_positions = [(m.start(), m.group(1)) for m in slide_markers]
        
        if not slide_positions:
            # Fall back to regular chunking if no slide markers
            return self._chunk_regular_document(document)
        
        # Process each slide as a chunk
        for i in range(len(slide_positions)):
            start_pos = slide_positions[i][0]
            slide_num = slide_positions[i][1]
            
            # Determine end position
            if i < len(slide_positions) - 1:
                end_pos = slide_positions[i+1][0]
            else:
                end_pos = len(text)
                
            # Extract slide text
            slide_text = text[start_pos:end_pos].strip()
            
            # Create chunk for slide
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=document.doc_id,
                text_chunk=slide_text,
                tag_context=document.persuasion_tags + [f"slide_{slide_num}"]
            )
            chunks.append(chunk.__dict__)
        
        # Update document with chunks
        document.chunks = chunks
        return document