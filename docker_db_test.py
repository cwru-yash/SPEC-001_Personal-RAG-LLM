import os
import sys
import requests

def test_duckdb_connection():
    """Test basic DuckDB connection."""
    try:
        import duckdb
        
        # First test in-memory connection
        conn_memory = duckdb.connect(":memory:")
        result = conn_memory.execute("SELECT 1 as test").fetchall()
        print(f"DuckDB in-memory connection successful")
        
        # Then test file connection
        db_path = "/data/metadata.db"
        print(f"Attempting to connect to {db_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = duckdb.connect(db_path)
        result = conn.execute("SELECT 1 as test").fetchall()
        
        print(f"DuckDB file connection successful: {result[0][0] == 1}")
        return True
    except Exception as e:
        print(f"DuckDB connection failed: {e}")
        return False

def test_neo4j_connection():
    """Test Neo4j connection."""
    try:
        from neo4j import GraphDatabase
        
        # Note the hostname uses the Docker service name
        uri = "bolt://neo4j:7687"
        user = "neo4j"
        password = "password"
        
        # Create a driver instance
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Verify connectivity
        with driver.session() as session:
            result = session.run("RETURN 1 as test").single()
            value = result["test"]
        
        driver.close()
        print(f"Neo4j connection successful: {value == 1}")
        return True
    except Exception as e:
        print(f"Neo4j connection failed: {e}")
        return False

def test_chroma_connection_direct():
    """Test ChromaDB connection using direct HTTP."""
    try:
        import requests
        
        # Test the v2 heartbeat endpoint
        response = requests.get("http://chroma:8000/api/v2/heartbeat")
        response.raise_for_status()
        heartbeat_data = response.json()
        print(f"ChromaDB API v2 heartbeat successful: {heartbeat_data}")
        
        # Test creating a collection using v2 API
        collection_name = "test_collection"
        
        # Check if collection exists
        response = requests.get("http://chroma:8000/api/v2/collections")
        response.raise_for_status()
        collections_data = response.json()
        
        # Create a test collection if it doesn't exist
        collection_exists = any(collection.get("name") == collection_name 
                              for collection in collections_data.get("collections", []))
        
        if not collection_exists:
            create_data = {
                "name": collection_name,
                "metadata": {"description": "Test collection"}
            }
            response = requests.post("http://chroma:8000/api/v2/collections", json=create_data)
            response.raise_for_status()
            print(f"Created test collection: {collection_name}")
        else:
            print(f"Test collection {collection_name} already exists")
        
        # List collections to verify
        response = requests.get("http://chroma:8000/api/v2/collections")
        response.raise_for_status()
        collections_data = response.json()
        collection_names = [col.get("name") for col in collections_data.get("collections", [])]
        
        print(f"Available collections: {collection_names}")
        return True
    except Exception as e:
        print(f"ChromaDB connection failed with direct HTTP: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chroma_connection():
    """Test ChromaDB connection."""
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Note: Since we're having issues with the client library,
        # use direct HTTP requests instead
        return test_chroma_connection_direct()
        
    except Exception as e:
        print(f"ChromaDB connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing database connections from inside Docker container...")
    
    # Test each database connection
    duckdb_ok = test_duckdb_connection()
    neo4j_ok = test_neo4j_connection()
    chroma_ok = test_chroma_connection()
    
    # Report overall status
    if duckdb_ok and neo4j_ok and chroma_ok:
        print("\n✅ All database connections successful!")
    else:
        print("\n❌ Some database connections failed.")
        if not duckdb_ok:
            print("  - DuckDB connection failed")
        if not neo4j_ok:
            print("  - Neo4j connection failed")
        if not chroma_ok:
            print("  - ChromaDB connection failed")