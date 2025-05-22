#!/usr/bin/env python3
"""
Initialize Neo4j database for the Personal RAG-LLM system.
"""
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
        with driver.session() as session:
            # Check if database exists
            db_result = session.run("SHOW DATABASES")
            databases = [record["name"] for record in db_result]
            
            if "documents" not in databases:
                print("Creating 'documents' database...")
                
                # Connect to system database to create a new database
                with driver.session(database="system") as system_session:
                    system_session.run("CREATE DATABASE documents IF NOT EXISTS")
                    print("Database 'documents' created successfully")
            else:
                print("Database 'documents' already exists")
    except Exception as e:
        print(f"Error creating database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    init_neo4j()