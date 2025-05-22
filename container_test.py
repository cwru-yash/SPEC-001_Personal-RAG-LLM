#!/usr/bin/env python3
"""
Simple test script to verify the RAG system components inside the Docker container.
"""
import os
import uuid
from datetime import datetime
from neo4j import GraphDatabase

def init_neo4j():
    """Initialize Neo4j database."""
    print("Initializing Neo4j database...")
    
    # Connect to Neo4j
    uri = "bolt://neo4j:7687"
    user = "neo4j"
    password = "password"
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # Create the documents database
    try:
        # Connect to system database to create a new database
        with driver.session(database="system") as system_session:
            system_session.run("CREATE DATABASE documents IF NOT EXISTS")
            print("Database 'documents' created (or already exists)")
            
            # Wait for the database to become available
            print("Waiting for database to be ready...")
            import time
            time.sleep(5)  # Give Neo4j time to create the database
    except Exception as e:
        print(f"Error initializing Neo4j: {e}")
    finally:
        driver.close()

def main():
    """Run a basic test of the system components."""
    print("=== Personal RAG-LLM System Test ===")
    
    # Initialize Neo4j database first
    init_neo4j()
    
    # Import models after database initialization
    from src.models.document import Document
    from src.storage.duckdb import DuckDBStorage
    from src.storage.vector_store import VectorStore
    from src.storage.graph_db import GraphDBStorage
    
    # Create test document
    doc_id = str(uuid.uuid4())
    test_doc = Document(
        doc_id=doc_id,
        file_name="test_document.pdf",
        file_extension="pdf",
        content_type=["pdf", "document"],
        text_content="This is a test document for the Personal RAG-LLM system.",
        metadata={
            "test": True,
            "created_by": "test_script"
        },
        created_at=datetime.now(),
        chunks=[{
            "chunk_id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "text_chunk": "This is a test document for the Personal RAG-LLM system.",
            "tag_context": ["test"]
        }]
    )
    
    print(f"Created test document with ID: {doc_id}")
    
    # Test DuckDB storage
    print("\n=== Testing DuckDB storage ===")
    try:
        db_path = "/data/metadata.db"
        print(f"Connecting to DuckDB at {db_path}")
        duckdb_storage = DuckDBStorage(db_path)
        
        print("Storing document...")
        result = duckdb_storage.store_document(test_doc)
        print(f"Storage result: {result}")
        
        print("Retrieving document...")
        retrieved_doc = duckdb_storage.get_document(doc_id)
        if retrieved_doc:
            print(f"Retrieved document: {retrieved_doc['file_name']}")
        else:
            print("Failed to retrieve document")
        
        print("DuckDB test completed successfully!")
    except Exception as e:
        print(f"DuckDB test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Neo4j storage
    print("\n=== Testing Neo4j Graph Database ===")
    try:
        graph_config = {
            "uri": "bolt://neo4j:7687",
            "user": "neo4j",
            "password": "password",
            "database": "documents"
        }
        print(f"Connecting to Neo4j at {graph_config['uri']}")
        graph_storage = GraphDBStorage(graph_config)
        
        print("Storing document in graph DB...")
        result = graph_storage.store_document(test_doc.__dict__)
        print(f"Storage result: {result}")
        
        print("Finding related documents...")
        related_docs = graph_storage.find_related_documents(doc_id)
        print(f"Found {len(related_docs)} related documents")
        
        print("Neo4j test completed successfully!")
    except Exception as e:
        print(f"Neo4j test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Vector Store
    print("\n=== Testing Vector Store ===")
    try:
        vector_config = {
            "storage_dir": "/data/vector_store"
        }
        print("Initializing vector store...")
        vector_store = VectorStore(vector_config)
        
        print("Storing document chunks...")
        result = vector_store.store_chunks(test_doc.chunks)
        print(f"Storage result: {result}")
        
        print("Searching for similar chunks...")
        search_results = vector_store.search("test document")
        print(f"Found {len(search_results)} similar chunks")
        
        print("Vector store test completed successfully!")
    except Exception as e:
        print(f"Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== All tests completed ===")

if __name__ == "__main__":
    main()