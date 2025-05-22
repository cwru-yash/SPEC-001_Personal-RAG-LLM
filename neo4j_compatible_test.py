#!/usr/bin/env python3
"""
Test script compatible with Neo4j Community Edition
"""
import os
import uuid
from datetime import datetime
from neo4j import GraphDatabase

def test_neo4j_connection():
    """Test Neo4j connection with the default database."""
    print("Testing Neo4j connection...")
    
    # Connect to Neo4j
    uri = "bolt://neo4j:7687"
    user = "neo4j"
    password = "password"
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        # Test connection with default database
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"Connected successfully! Found {count} nodes in the database.")
        return True
    except Exception as e:
        print(f"Connection error: {e}")
        return False
    finally:
        driver.close()

def modify_graph_db_for_default_database():
    """Create a version of GraphDBStorage that uses the default database."""
    # Path to the original graph_db.py
    original_path = "/app/src/storage/graph_db.py"
    
    # Read the original file
    with open(original_path, "r") as f:
        content = f.read()
    
    # Replace the database name with "neo4j" (the default)
    modified = content.replace('self.database = config.get("database", "documents")', 
                             'self.database = config.get("database", "neo4j")')
    
    # Write the modified file
    with open(original_path, "w") as f:
        f.write(modified)
    
    print("Updated GraphDBStorage to use the default database.")

def main():
    """Run a basic test of the system components."""
    print("=== Personal RAG-LLM System Test (Neo4j Community Compatible) ===")
    
    # Test Neo4j connection first
    if not test_neo4j_connection():
        print("Failed to connect to Neo4j. Please check your Neo4j configuration.")
        return

    # Update GraphDBStorage to use default database
    try:
        modify_graph_db_for_default_database()
    except Exception as e:
        print(f"Error updating GraphDBStorage: {e}")
        return
    
    # Import models after database check
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
        
        print("DuckDB test completed!")
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
            "database": "neo4j"  # Use default database
        }
        print(f"Connecting to Neo4j at {graph_config['uri']}")
        graph_storage = GraphDBStorage(graph_config)
        
        print("Storing document in graph DB...")
        result = graph_storage.store_document(test_doc.__dict__)
        print(f"Storage result: {result}")
        
        # Skip related documents test for simplicity
        
        print("Neo4j test completed!")
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
        
        print("Vector store test completed!")
    except Exception as e:
        print(f"Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== All tests completed ===")

if __name__ == "__main__":
    main()