import os
import sys
import uuid

# When running inside the container, the code is directly in /app/src
from src.models.document import Document
from src.storage.duckdb import DuckDBStorage
from src.storage.vector_store import VectorStore
from src.storage.graph_db import GraphDBStorage

def main():
    """Test the integration of all components inside Docker."""
    print("Starting Docker container integration test")
    
    # Create a test document
    doc_id = str(uuid.uuid4())
    test_doc = Document(
        doc_id=doc_id,
        file_name="test_doc.pdf",
        file_extension="pdf",
        content_type=["pdf", "document"],
        text_content="This is a test document for the Personal RAG-LLM system.",
        metadata={"test": True},
        chunks=[{
            "chunk_id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "text_chunk": "This is a test document.",
            "tag_context": ["test"]
        }]
    )
    
    print(f"Created test document with ID: {doc_id}")
    
    # Create config dictionaries for services
    duckdb_config = {
        "database": "/data/metadata.db"
    }
    
    vector_store_config = {
        "host": "chroma",  # Docker service name
        "port": 8000,
        "embedding_function": "sentence-transformers/all-MiniLM-L6-v2",
        "collections": {
            "document_chunks": {
                "embedding_function": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384
            }
        }
    }
    
    graph_db_config = {
        "uri": "bolt://neo4j:7687",  # Docker service name
        "user": "neo4j",
        "password": "password",
        "database": "documents"
    }
    
    # Initialize storage components
    try:
        print("Initializing DuckDB storage...")
        duckdb_storage = DuckDBStorage(duckdb_config["database"])
        print("DuckDB connection established!")
        
        print("Testing document storage in DuckDB...")
        duckdb_result = duckdb_storage.store_document(test_doc)
        print(f"DuckDB storage result: {duckdb_result}")
        
        print("\nInitializing Graph DB storage...")
        graph_storage = GraphDBStorage(graph_db_config)
        print("Graph DB connection established!")
        
        print("Testing document storage in Graph DB...")
        graph_result = graph_storage.store_document(test_doc.__dict__)
        print(f"Graph DB storage result: {graph_result}")
        
        print("\nInitializing Vector Store...")
        vector_store = VectorStore(vector_store_config)
        print("Vector Store connection established!")
        
        print("Testing chunk storage in Vector DB...")
        vector_result = vector_store.store_chunks(test_doc.chunks)
        print(f"Vector store result: {vector_result}")
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()