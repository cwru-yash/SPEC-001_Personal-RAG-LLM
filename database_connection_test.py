import os
import sys
import uuid

# Add the project root to path
sys.path.append('services/processor')

def test_duckdb_connection():
    """Test basic DuckDB connection."""
    try:
        import duckdb
        
        # Connect to a temporary in-memory database
        conn = duckdb.connect(":memory:")
        
        # Execute a simple query
        result = conn.execute("SELECT 1 as test").fetchall()
        
        print(f"DuckDB connection successful: {result[0][0] == 1}")
        return True
    except Exception as e:
        print(f"DuckDB connection failed: {e}")
        return False

def test_neo4j_connection():
    """Test Neo4j connection."""
    try:
        from neo4j import GraphDatabase
        
        # These should match your docker-compose settings
        uri = "bolt://localhost:7687"
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

def test_chroma_connection():
    """Test ChromaDB connection."""
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Connect to ChromaDB
        client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            settings=Settings(
                chroma_api_impl="rest",
                chroma_server_host="localhost",
                chroma_server_http_port=8000
            )
        )
        
        # Check connection by getting collections
        collections = client.list_collections()
        
        print(f"ChromaDB connection successful: {client is not None}")
        print(f"Available collections: {[c.name for c in collections]}")
        return True
    except Exception as e:
        print(f"ChromaDB connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing database connections...")
    
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