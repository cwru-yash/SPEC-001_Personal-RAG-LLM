# src/storage/vector_store.py
import os
import json
from typing import Dict, List, Any, Optional
import uuid
import numpy as np
import tempfile

class VectorStore:
    """A simplified vector store implementation using local file storage.
    
    This is a temporary solution for testing until the ChromaDB integration
    is fully working. It stores vectors in a local JSON file.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector store."""
        self.config = config
        self.storage_dir = "/data/vector_store"
        self.index_file = os.path.join(self.storage_dir, "index.json")
        
        # Create the storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize the index if it doesn't exist
        if not os.path.exists(self.index_file):
            self._initialize_index()
    
    def _initialize_index(self):
        """Create a new index file."""
        index = {
            "chunks": {},
            "metadata": {
                "created_at": self._get_current_time(),
                "version": "1.0"
            }
        }
        with open(self.index_file, 'w') as f:
            json.dump(index, f)
        print(f"Initialized vector store at {self.index_file}")
    
    def _load_index(self):
        """Load the index from disk."""
        try:
            with open(self.index_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file doesn't exist or is corrupted, create a new one
            self._initialize_index()
            return self._load_index()
    
    def _save_index(self, index):
        """Save the index to disk."""
        # First save to a temporary file, then rename it
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=self.storage_dir) as temp_file:
            json.dump(index, temp_file)
            temp_filename = temp_file.name
        
        # Rename the temporary file to the actual index file
        os.replace(temp_filename, self.index_file)
    
    def _get_current_time(self):
        """Get current time for logging."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _create_embedding(self, text):
        """Create a simple embedding for a text string.
        
        In a real implementation, this would use a text embedding model.
        Here we're using a simplified hash-based approach for testing.
        """
        import hashlib
        
        # Create a deterministic but simple "embedding" by hashing
        h = hashlib.md5(text.encode()).digest()
        
        # Convert 16 bytes to 64 floats (repeating the pattern)
        embedding = []
        for _ in range(4):
            for b in h:
                embedding.append(float(b) / 255.0)
        
        return embedding
    
    def store_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Store document chunk in vector database."""
        try:
            chunk_id = chunk["chunk_id"]
            doc_id = chunk["doc_id"]
            text_chunk = chunk["text_chunk"]
            tag_context = chunk.get("tag_context", [])
            
            # Create metadata and embedding
            metadata = {
                "doc_id": doc_id,
                "tag_context": tag_context if isinstance(tag_context, list) else [tag_context],
                "created_at": self._get_current_time()
            }
            embedding = self._create_embedding(text_chunk)
            
            # Load the current index
            index = self._load_index()
            
            # Add the chunk
            index["chunks"][chunk_id] = {
                "text": text_chunk,
                "metadata": metadata,
                "embedding": embedding
            }
            
            # Save the updated index
            self._save_index(index)
            
            return True
        except Exception as e:
            print(f"Error storing chunk in vector store: {e}")
            return False
    
    def store_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Store multiple document chunks in vector database."""
        try:
            # Load the current index
            index = self._load_index()
            
            # Add each chunk
            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                doc_id = chunk["doc_id"]
                text_chunk = chunk["text_chunk"]
                tag_context = chunk.get("tag_context", [])
                
                # Create metadata and embedding
                metadata = {
                    "doc_id": doc_id,
                    "tag_context": tag_context if isinstance(tag_context, list) else [tag_context],
                    "created_at": self._get_current_time()
                }
                embedding = self._create_embedding(text_chunk)
                
                # Add the chunk to the index
                index["chunks"][chunk_id] = {
                    "text": text_chunk,
                    "metadata": metadata,
                    "embedding": embedding
                }
            
            # Save the updated index
            self._save_index(index)
            
            return True
        except Exception as e:
            print(f"Error storing chunks in vector store: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search(self, query: str, top_k: int = 5, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for relevant document chunks using semantic similarity."""
        try:
            # Create a query embedding
            query_embedding = self._create_embedding(query)
            
            # Load the index
            index = self._load_index()
            
            # Calculate similarity for each chunk
            results = []
            for chunk_id, chunk_data in index["chunks"].items():
                # Apply filter if provided
                if filter_dict:
                    # Simple filtering - check if filter keys exist in metadata
                    metadata = chunk_data["metadata"]
                    matches_filter = True
                    for key, value in filter_dict.items():
                        if key not in metadata or metadata[key] != value:
                            matches_filter = False
                            break
                    
                    if not matches_filter:
                        continue
                
                # Calculate cosine similarity
                chunk_embedding = chunk_data["embedding"]
                similarity = self._calculate_similarity(query_embedding, chunk_embedding)
                
                results.append({
                    "chunk_id": chunk_id,
                    "doc_id": chunk_data["metadata"]["doc_id"],
                    "text": chunk_data["text"],
                    "metadata": chunk_data["metadata"],
                    "similarity": similarity
                })
            
            # Sort by similarity (descending) and limit to top_k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            results = results[:top_k]
            
            return results
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def _calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings."""
        # Convert to numpy arrays
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def delete_document_chunks(self, doc_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            # Load the index
            index = self._load_index()
            
            # Find chunks to delete
            chunks_to_delete = []
            for chunk_id, chunk_data in index["chunks"].items():
                if chunk_data["metadata"]["doc_id"] == doc_id:
                    chunks_to_delete.append(chunk_id)
            
            # Delete chunks
            for chunk_id in chunks_to_delete:
                del index["chunks"][chunk_id]
            
            # Save the updated index
            self._save_index(index)
            
            return True
        except Exception as e:
            print(f"Error deleting document chunks: {e}")
            return False