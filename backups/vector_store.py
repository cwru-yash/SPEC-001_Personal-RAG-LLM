# src/storage/vector_store.py
import os
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid

from src.models.document import DocumentChunk

class VectorStore:
    """Storage interface for vector database to enable semantic search."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ChromaDB client and collections."""
        self.config = config
        self.host = config.get("host", "chroma")
        self.port = config.get("port", 8000)
        
        # Setup ChromaDB client
        self.client = chromadb.HttpClient(
            host=self.host,
            port=self.port,
            settings=Settings(
                chroma_api_impl="rest",
                chroma_server_host=self.host,
                chroma_server_http_port=self.port
            )
        )
        
        # Initialize the embedding function
        self.embedding_model_name = config.get("embedding_function", 
                                              "sentence-transformers/all-MiniLM-L6-v2")
        
        # Create collections
        self._initialize_collections()
        
        # Initialize local embedding model for when direct API calls aren't possible
        # This is a fallback when the ChromaDB server embedding function isn't available
        self.local_embedder = None  # Lazy-loaded when needed
    
    def _initialize_collections(self):
        """Create necessary collections if they don't exist."""
        # Default collections from config
        collections_config = self.config.get("collections", {})
        
        # Ensure required collections exist
        required_collections = ["document_chunks", "query_history"]
        
        # Set up sentence transformer embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
        
        # Create each collection
        for collection_name in required_collections:
            try:
                # Try to get the collection if it exists
                self.client.get_collection(name=collection_name)
            except Exception:
                # Create collection if it doesn't exist
                self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": f"Collection for {collection_name}"}
                )
    
    def _get_local_embedder(self):
        """Get or initialize local embedder for fallback."""
        if self.local_embedder is None:
            self.local_embedder = SentenceTransformer(self.embedding_model_name)
        return self.local_embedder
    
    def store_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Store document chunk in vector database."""
        try:
            # Get document chunks collection
            collection = self.client.get_collection(
                name="document_chunks",
                embedding_function=self.embedding_function
            )
            
            # Extract text and metadata
            chunk_id = chunk["chunk_id"]
            doc_id = chunk["doc_id"]
            text_chunk = chunk["text_chunk"]
            tag_context = chunk.get("tag_context", [])
            
            # Convert tag_context to comma-separated string if it's a list
            if isinstance(tag_context, list):
                tag_context = ", ".join(tag_context)
            
            # Create metadata dictionary
            metadata = {
                "doc_id": doc_id,
                "tag_context": tag_context
            }
            
            # Add document to collection
            collection.add(
                ids=[chunk_id],
                documents=[text_chunk],
                metadatas=[metadata]
            )
            
            return True
            
        except Exception as e:
            print(f"Error storing chunk in vector DB: {e}")
            return False
    
    def store_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Store multiple document chunks in vector database."""
        try:
            # Get document chunks collection
            collection = self.client.get_collection(
                name="document_chunks",
                embedding_function=self.embedding_function
            )
            
            # Prepare data for batch insertion
            ids = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                doc_id = chunk["doc_id"]
                text_chunk = chunk["text_chunk"]
                tag_context = chunk.get("tag_context", [])
                
                # Convert tag_context to comma-separated string if it's a list
                if isinstance(tag_context, list):
                    tag_context = ", ".join(tag_context)
                
                # Create metadata dictionary
                metadata = {
                    "doc_id": doc_id,
                    "tag_context": tag_context
                }
                
                ids.append(chunk_id)
                documents.append(text_chunk)
                metadatas.append(metadata)
            
            # Add documents to collection in batch
            if ids:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            return True
            
        except Exception as e:
            print(f"Error storing chunks in vector DB: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for relevant document chunks using semantic similarity."""
        try:
            # Get document chunks collection
            collection = self.client.get_collection(
                name="document_chunks",
                embedding_function=self.embedding_function
            )
            
            # Prepare filter if provided
            where_filter = {}
            if filter_dict:
                for key, value in filter_dict.items():
                    where_filter[key] = value
            
            # Search for similar documents
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    chunk_id = results["ids"][0][i]
                    text = results["documents"][0][i]
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    
                    formatted_results.append({
                        "chunk_id": chunk_id,
                        "doc_id": metadata.get("doc_id", ""),
                        "text": text,
                        "metadata": metadata,
                        "similarity": 1 - distance  # Convert distance to similarity score
                    })
            
            # Log the query for future analysis
            self._log_query(query, formatted_results)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching vector DB: {e}")
            return []
    
    def delete_document_chunks(self, doc_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            # Get document chunks collection
            collection = self.client.get_collection(
                name="document_chunks",
                embedding_function=self.embedding_function
            )
            
            # Delete all chunks for the given document
            collection.delete(
                where={"doc_id": doc_id}
            )
            
            return True
            
        except Exception as e:
            print(f"Error deleting document chunks: {e}")
            return False
    
    def _log_query(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Log search query for analytics and improvement."""
        try:
            # Get query history collection
            collection = self.client.get_collection(
                name="query_history",
                embedding_function=self.embedding_function
            )
            
            # Generate unique ID for query
            query_id = str(uuid.uuid4())
            
            # Create metadata with result summary
            result_docs = [result["doc_id"] for result in results]
            result_scores = [result["similarity"] for result in results]
            
            metadata = {
                "query_time": str(self._get_current_time()),
                "result_count": len(results),
                "result_docs": ", ".join(result_docs) if result_docs else "",
                "avg_similarity": sum(result_scores) / len(result_scores) if result_scores else 0
            }
            
            # Store query in history
            collection.add(
                ids=[query_id],
                documents=[query],
                metadatas=[metadata]
            )
            
        except Exception as e:
            print(f"Error logging query: {e}")
    
    def _get_current_time(self):
        """Get current time for logging."""
        from datetime import datetime
        return datetime.now()