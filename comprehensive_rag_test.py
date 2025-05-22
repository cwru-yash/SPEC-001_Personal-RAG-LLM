#!/usr/bin/env python3
"""
Comprehensive test for the Personal RAG-LLM system.
Tests all components and their integration.
"""
import os
import uuid
import json
import time
from datetime import datetime
import requests
import numpy as np
from neo4j import GraphDatabase
import duckdb

class RAGSystemTester:
    """Test all components of the RAG system."""
    
    def __init__(self):
        """Initialize the test environment."""
        self.test_id = str(uuid.uuid4())[:8]
        self.results = {
            "services": {},
            "storage": {},
            "integration": {},
            "summary": {
                "passed": 0,
                "failed": 0,
                "total": 0
            }
        }
        self.start_time = time.time()
        
        # Import components
        try:
            from src.models.document import Document
            from src.storage.duckdb import DuckDBStorage
            from src.storage.vector_store import VectorStore
            from src.storage.graph_db import GraphDBStorage
            
            self.Document = Document
            self.DuckDBStorage = DuckDBStorage
            self.VectorStore = VectorStore
            self.GraphDBStorage = GraphDBStorage
            self.imports_success = True
        except Exception as e:
            print(f"Error importing components: {e}")
            self.imports_success = False
    
    def record_result(self, category, test_name, passed, message=None, details=None):
        """Record a test result."""
        if category not in self.results:
            self.results[category] = {}
            
        self.results[category][test_name] = {
            "passed": passed,
            "message": message or ("PASSED" if passed else "FAILED"),
            "details": details
        }
        
        # Update summary
        self.results["summary"]["total"] += 1
        if passed:
            self.results["summary"]["passed"] += 1
        else:
            self.results["summary"]["failed"] += 1
        
        # Print result
        status = "✅" if passed else "❌"
        print(f"{status} {category} - {test_name}: {message or ('PASSED' if passed else 'FAILED')}")
    
    def test_services(self):
        """Test if all required services are running."""
        print("\n=== Testing Services ===")
        
        # Test DuckDB
        try:
            conn = duckdb.connect(":memory:")
            result = conn.execute("SELECT 1 as test").fetchall()
            self.record_result("services", "duckdb", True, "DuckDB service is available")
        except Exception as e:
            self.record_result("services", "duckdb", False, f"DuckDB service error: {e}")
        
        # Test Neo4j
        try:
            uri = "bolt://neo4j:7687"
            driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))
            with driver.session() as session:
                result = session.run("RETURN 1 as test").single()
                self.record_result("services", "neo4j", True, "Neo4j service is available")
            driver.close()
        except Exception as e:
            self.record_result("services", "neo4j", False, f"Neo4j service error: {e}")
        
        # Test API service if available
        try:
            response = requests.get("http://api:4000/", timeout=2)
            status = response.status_code
            self.record_result("services", "api", status == 200, 
                               f"API service returned status {status}")
        except Exception as e:
            self.record_result("services", "api", False, 
                               "API service is not available (this is OK if you're not testing the API)")
    
    def test_storage_components(self):
        """Test storage components individually."""
        if not self.imports_success:
            self.record_result("storage", "imports", False, "Failed to import required components")
            return
            
        print("\n=== Testing Storage Components ===")
        
        # Create test document
        try:
            doc_id = f"test-doc-{self.test_id}"
            test_doc = self.Document(
                doc_id=doc_id,
                file_name=f"test-{self.test_id}.pdf",
                file_extension="pdf",
                content_type=["pdf", "document"],
                text_content=f"This is a test document {self.test_id} for RAG testing.",
                metadata={
                    "test_id": self.test_id,
                    "created_at": datetime.now().isoformat()
                },
                chunks=[{
                    "chunk_id": f"chunk-{self.test_id}-1",
                    "doc_id": doc_id,
                    "text_chunk": f"Test chunk 1 for document {self.test_id}",
                    "tag_context": ["test", "chunk1"]
                }, {
                    "chunk_id": f"chunk-{self.test_id}-2",
                    "doc_id": doc_id,
                    "text_chunk": f"Test chunk 2 for document {self.test_id}",
                    "tag_context": ["test", "chunk2"]
                }]
            )
            self.record_result("storage", "document_creation", True, "Test document created")
            self.test_doc = test_doc
        except Exception as e:
            self.record_result("storage", "document_creation", False, f"Document creation error: {e}")
            return
        
        # Test DuckDB Storage
        try:
            db_path = "/data/metadata.db"
            duckdb_storage = self.DuckDBStorage(db_path)
            result = duckdb_storage.store_document(self.test_doc)
            self.record_result("storage", "duckdb_store", result, 
                               "Document stored in DuckDB" if result else "Failed to store document in DuckDB")
            
            if result:
                retrieved = duckdb_storage.get_document(doc_id)
                retrieval_success = retrieved is not None
                self.record_result("storage", "duckdb_retrieve", retrieval_success,
                                  "Document retrieved from DuckDB" if retrieval_success 
                                  else "Failed to retrieve document from DuckDB")
            
            self.duckdb_storage = duckdb_storage
        except Exception as e:
            self.record_result("storage", "duckdb", False, f"DuckDB error: {e}")
        
        # Test Neo4j Graph DB
        try:
            graph_config = {
                "uri": "bolt://neo4j:7687",
                "user": "neo4j",
                "password": "password",
                "database": "neo4j"  # Use default database
            }
            graph_storage = self.GraphDBStorage(graph_config)
            result = graph_storage.store_document(self.test_doc.__dict__)
            self.record_result("storage", "neo4j_store", result,
                              "Document stored in Neo4j" if result 
                              else "Failed to store document in Neo4j")
            
            self.graph_storage = graph_storage
        except Exception as e:
            self.record_result("storage", "neo4j", False, f"Neo4j error: {e}")
        
        # Test Vector Store
        try:
            vector_config = {
                "storage_dir": "/data/vector_store"
            }
            vector_store = self.VectorStore(vector_config)
            result = vector_store.store_chunks(self.test_doc.chunks)
            self.record_result("storage", "vector_store", result,
                              "Chunks stored in vector store" if result 
                              else "Failed to store chunks in vector store")
            
            if result:
                search_results = vector_store.search("test chunk")
                search_success = len(search_results) > 0
                self.record_result("storage", "vector_search", search_success,
                                  f"Found {len(search_results)} chunks in vector search" 
                                  if search_success else "Vector search found no results")
            
            self.vector_store = vector_store
        except Exception as e:
            self.record_result("storage", "vector_store", False, f"Vector store error: {e}")
    
    def test_integration(self):
        """Test integration between components."""
        if not hasattr(self, 'test_doc'):
            self.record_result("integration", "prerequisites", False, 
                              "Cannot test integration without test document")
            return
            
        print("\n=== Testing Component Integration ===")
        
        # Test DuckDB + Vector Store integration
        try:
            # Fetch document from DuckDB
            if hasattr(self, 'duckdb_storage') and hasattr(self, 'vector_store'):
                doc_id = self.test_doc.doc_id
                retrieved_doc = self.duckdb_storage.get_document(doc_id)
                
                if retrieved_doc:
                    # Search for the document in vector store
                    search_text = retrieved_doc['text_content'][:20]  # Use part of text for search
                    search_results = self.vector_store.search(search_text)
                    
                    # Check if any results match the document ID
                    doc_found = any(r['doc_id'] == doc_id for r in search_results)
                    self.record_result("integration", "duckdb_vector_store", doc_found,
                                      "Successfully found document text in vector store" if doc_found
                                      else "Could not find document text in vector store")
                else:
                    self.record_result("integration", "duckdb_vector_store", False,
                                      "Could not retrieve document from DuckDB")
            else:
                self.record_result("integration", "duckdb_vector_store", False,
                                 "DuckDB or Vector Store not available")
        except Exception as e:
            self.record_result("integration", "duckdb_vector_store", False,
                              f"DuckDB + Vector Store integration error: {e}")
        
        # Test Neo4j + DuckDB integration
        try:
            if hasattr(self, 'graph_storage') and hasattr(self, 'duckdb_storage'):
                # Find related documents in Neo4j
                doc_id = self.test_doc.doc_id
                
                # Store a second document with a relationship
                related_doc_id = f"related-{self.test_id}"
                related_doc = self.Document(
                    doc_id=related_doc_id,
                    file_name=f"related-{self.test_id}.pdf",
                    file_extension="pdf",
                    content_type=["pdf", "document"],
                    text_content=f"This document is related to {doc_id}",
                    metadata={
                        "references": [{"doc_id": doc_id, "type": "REFERENCES"}]
                    }
                )
                
                # Store in both DuckDB and Neo4j
                self.duckdb_storage.store_document(related_doc)
                self.graph_storage.store_document(related_doc.__dict__)
                
                # Now check if we can find the relationship
                try:
                    related_docs = self.graph_storage.find_related_documents(doc_id)
                    found_relation = any(d.get('doc_id') == related_doc_id for d in related_docs)
                    self.record_result("integration", "neo4j_relationships", found_relation,
                                      "Successfully found document relationships in Neo4j" if found_relation
                                      else "Could not find document relationships in Neo4j")
                except Exception as e:
                    self.record_result("integration", "neo4j_relationships", False,
                                      f"Error finding related documents: {e}")
                    
            else:
                self.record_result("integration", "neo4j_duckdb", False,
                                 "Neo4j or DuckDB not available")
        except Exception as e:
            self.record_result("integration", "neo4j_duckdb", False,
                              f"Neo4j + DuckDB integration error: {e}")
    
    def test_persistence(self):
        """Test data persistence by retrieving previously stored data."""
        print("\n=== Testing Data Persistence ===")
        
        # Try to retrieve data stored in previous tests
        doc_id = f"test-doc-{self.test_id}"
        
        # Try to retrieve from DuckDB
        try:
            db_path = "/data/metadata.db"
            duckdb_storage = self.DuckDBStorage(db_path)
            retrieved_doc = duckdb_storage.get_document(doc_id)
            
            persistence_ok = retrieved_doc is not None
            self.record_result("persistence", "duckdb", persistence_ok,
                              "Successfully retrieved document from persistent storage" if persistence_ok
                              else "Could not retrieve document from persistent storage")
        except Exception as e:
            self.record_result("persistence", "duckdb", False,
                              f"DuckDB persistence error: {e}")
        
        # Try to retrieve from Vector Store
        try:
            vector_config = {"storage_dir": "/data/vector_store"}
            vector_store = self.VectorStore(vector_config)
            search_results = vector_store.search(f"document {self.test_id}")
            
            persistence_ok = len(search_results) > 0
            self.record_result("persistence", "vector_store", persistence_ok,
                              f"Successfully found {len(search_results)} chunks in persistent vector store" 
                              if persistence_ok else "Could not find data in persistent vector store")
        except Exception as e:
            self.record_result("persistence", "vector_store", False,
                              f"Vector store persistence error: {e}")
    
    def print_summary(self):
        """Print a summary of all test results."""
        print("\n=== Test Summary ===")
        print(f"Total tests: {self.results['summary']['total']}")
        print(f"Passed: {self.results['summary']['passed']}")
        print(f"Failed: {self.results['summary']['failed']}")
        
        if self.results['summary']['failed'] > 0:
            print("\nFailed tests:")
            for category in self.results:
                if category == "summary":
                    continue
                    
                for test_name, result in self.results[category].items():
                    if not result["passed"]:
                        print(f"  - {category}.{test_name}: {result['message']}")
        
        # Calculate pass percentage
        if self.results['summary']['total'] > 0:
            pass_percent = (self.results['summary']['passed'] / self.results['summary']['total']) * 100
            print(f"\nPass rate: {pass_percent:.1f}%")
        
        # Time elapsed
        elapsed = time.time() - self.start_time
        print(f"Test duration: {elapsed:.2f} seconds")
    
    def save_results(self):
        """Save test results to a file."""
        results_dir = "/data/test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\nTest results saved to {filename}")
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        print("Starting comprehensive RAG system tests...")
        
        if not self.imports_success:
            print("❌ Failed to import required components. Cannot proceed with tests.")
            return
        
        # Run all test phases
        self.test_services()
        self.test_storage_components()
        self.test_integration()
        self.test_persistence()
        
        # Print summary and save results
        self.print_summary()
        self.save_results()

if __name__ == "__main__":
    tester = RAGSystemTester()
    tester.run_all_tests()