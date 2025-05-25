# test_complete_rag_pipeline.py - End-to-end RAG system test
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

# Import storage components
from services.processor.src.storage.duckdb import DuckDBStorage
from services.processor.src.storage.vector_store import VectorStore
from services.processor.src.storage.graph_db import GraphDBStorage

class RAGPipelineTester:
    """Test the complete RAG pipeline from document storage to retrieval."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the tester with storage backends."""
        self.config = config
        self.metadata_storage = DuckDBStorage(config["storage"]["duckdb"]["database"])
        self.vector_store = VectorStore(config["storage"]["vector_store"])
        self.graph_storage = GraphDBStorage(config["storage"]["graph_db"])
        
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {"passed": 0, "failed": 0, "total": 0}
        }
    
    def record_test(self, test_name: str, passed: bool, details: Dict[str, Any] = None):
        """Record a test result."""
        self.test_results["tests"][test_name] = {
            "passed": passed,
            "details": details or {}
        }
        self.test_results["summary"]["total"] += 1
        if passed:
            self.test_results["summary"]["passed"] += 1
        else:
            self.test_results["summary"]["failed"] += 1
        
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")
    
    def test_storage_connectivity(self):
        """Test connectivity to all storage systems."""
        print("\n=== Testing Storage Connectivity ===")
        
        # Test DuckDB
        try:
            result = self.metadata_storage.conn.execute("SELECT 1").fetchone()
            self.record_test("DuckDB connectivity", True, {"status": "Connected"})
        except Exception as e:
            self.record_test("DuckDB connectivity", False, {"error": str(e)})
        
        # Test Vector Store (ChromaDB)
        try:
            # Try to list collections or do a dummy search
            results = self.vector_store.search("test", top_k=1)
            self.record_test("Vector Store connectivity", True, {"status": "Connected"})
        except Exception as e:
            self.record_test("Vector Store connectivity", False, {"error": str(e)})
        
        # Test Neo4j
        try:
            with self.graph_storage.driver.session() as session:
                result = session.run("RETURN 1 as test").single()
                self.record_test("Neo4j connectivity", True, {"status": "Connected"})
        except Exception as e:
            self.record_test("Neo4j connectivity", False, {"error": str(e)})
    
    def test_document_retrieval(self):
        """Test document retrieval from all storage systems."""
        print("\n=== Testing Document Retrieval ===")
        
        # Get sample documents from DuckDB
        try:
            docs = self.metadata_storage.search_documents({"text_search": ""})[:5]
            self.record_test("DuckDB document search", True, 
                           {"documents_found": len(docs)})
            
            if docs:
                # Test retrieving a specific document
                test_doc = self.metadata_storage.get_document(docs[0]["doc_id"])
                self.record_test("DuckDB document retrieval", test_doc is not None,
                               {"doc_id": docs[0]["doc_id"] if test_doc else None})
        except Exception as e:
            self.record_test("DuckDB document operations", False, {"error": str(e)})
    
    def test_semantic_search(self):
        """Test semantic search functionality."""
        print("\n=== Testing Semantic Search ===")
        
        test_queries = [
            "data analysis",
            "machine learning",
            "email communication",
            "spreadsheet data",
            "presentation slides"
        ]
        
        for query in test_queries:
            try:
                results = self.vector_store.search(query, top_k=3)
                self.record_test(f"Semantic search: '{query}'", True, {
                    "results_found": len(results),
                    "top_similarity": results[0]["similarity"] if results else 0
                })
            except Exception as e:
                self.record_test(f"Semantic search: '{query}'", False, {"error": str(e)})
    
    def test_graph_relationships(self):
        """Test graph relationship queries."""
        print("\n=== Testing Graph Relationships ===")
        
        try:
            # Get a sample document
            docs = self.metadata_storage.search_documents({})[:1]
            
            if docs:
                doc_id = docs[0]["doc_id"]
                
                # Find related documents
                related = self.graph_storage.find_related_documents(doc_id)
                self.record_test("Graph relationship query", True, {
                    "source_doc": doc_id,
                    "related_found": len(related)
                })
                
                # Get document graph
                graph = self.graph_storage.get_document_graph(doc_id)
                self.record_test("Document graph retrieval", True, {
                    "nodes": len(graph.get("nodes", [])),
                    "edges": len(graph.get("edges", []))
                })
            else:
                self.record_test("Graph operations", False, 
                               {"error": "No documents found for testing"})
                
        except Exception as e:
            self.record_test("Graph operations", False, {"error": str(e)})
    
    def test_category_specific_queries(self):
        """Test queries specific to each category."""
        print("\n=== Testing Category-Specific Queries ===")
        
        categories = ["Documents", "Emails", "Images", "Presentations", "Spreadsheets"]
        
        for category in categories:
            try:
                # Search by category
                docs = self.metadata_storage.search_documents({
                    "text_search": category.lower()
                })
                
                # Count documents in this category
                category_count = sum(1 for doc in docs 
                                   if category.lower() in str(doc.get("content_type", [])).lower())
                
                self.record_test(f"Category search: {category}", True, {
                    "total_found": len(docs),
                    "category_matches": category_count
                })
                
            except Exception as e:
                self.record_test(f"Category search: {category}", False, {"error": str(e)})
    
    def test_paired_document_relationships(self):
        """Test relationships between paired documents."""
        print("\n=== Testing Paired Document Relationships ===")
        
        try:
            # Query for DESCRIBES relationships in Neo4j
            with self.graph_storage.driver.session() as session:
                result = session.run("""
                    MATCH (m:MetadataDocument)-[r:DESCRIBES]->(d:ResearchDocument)
                    RETURN count(r) as pair_count
                    LIMIT 1
                """)
                
                record = result.single()
                if record:
                    pair_count = record["pair_count"]
                    self.record_test("Paired document relationships", True, {
                        "pairs_found": pair_count
                    })
                else:
                    self.record_test("Paired document relationships", False, {
                        "error": "No paired relationships found"
                    })
                    
        except Exception as e:
            self.record_test("Paired document relationships", False, {"error": str(e)})
    
    def test_rag_query_pipeline(self):
        """Test a complete RAG query pipeline."""
        print("\n=== Testing Complete RAG Query Pipeline ===")
        
        test_query = "What information do we have about data analysis?"
        
        try:
            # Step 1: Semantic search
            semantic_results = self.vector_store.search(test_query, top_k=5)
            
            if semantic_results:
                # Step 2: Get full documents
                doc_ids = [r["doc_id"] for r in semantic_results]
                documents = []
                
                for doc_id in doc_ids:
                    doc = self.metadata_storage.get_document(doc_id)
                    if doc:
                        documents.append(doc)
                
                # Step 3: Find related documents via graph
                all_related = []
                for doc_id in doc_ids[:2]:  # Check first 2
                    related = self.graph_storage.find_related_documents(doc_id, max_depth=1)
                    all_related.extend(related)
                
                self.record_test("Complete RAG query pipeline", True, {
                    "query": test_query,
                    "semantic_results": len(semantic_results),
                    "documents_retrieved": len(documents),
                    "related_documents": len(all_related),
                    "total_context": len(documents) + len(all_related)
                })
                
                # Print sample result
                if documents:
                    print(f"\n    Sample retrieved content:")
                    sample_text = documents[0].get("text_content", "")[:200]
                    print(f"    {sample_text}...")
                    
            else:
                self.record_test("Complete RAG query pipeline", False, {
                    "error": "No semantic search results found"
                })
                
        except Exception as e:
            self.record_test("Complete RAG query pipeline", False, {"error": str(e)})
    
    def generate_report(self):
        """Generate a test report."""
        print("\n" + "="*60)
        print("RAG PIPELINE TEST REPORT")
        print("="*60)
        
        summary = self.test_results["summary"]
        success_rate = (summary["passed"] / summary["total"] * 100 
                       if summary["total"] > 0 else 0)
        
        print(f"Total Tests: {summary['total']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if summary["failed"] > 0:
            print("\nFailed Tests:")
            for test_name, result in self.test_results["tests"].items():
                if not result["passed"]:
                    print(f"  - {test_name}")
                    if "error" in result["details"]:
                        print(f"    Error: {result['details']['error']}")
        
        # Save report
        report_file = f"rag_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        print("Starting RAG Pipeline Tests...")
        print("="*60)
        
        self.test_storage_connectivity()
        self.test_document_retrieval()
        self.test_semantic_search()
        self.test_graph_relationships()
        self.test_category_specific_queries()
        self.test_paired_document_relationships()
        self.test_rag_query_pipeline()
        
        self.generate_report()


def query_rag_system(query: str, config: Dict[str, Any]):
    """Example function to query the RAG system."""
    print(f"\n🔍 RAG Query: {query}")
    print("-" * 40)
    
    # Initialize storage
    vector_store = VectorStore(config["storage"]["vector_store"])
    metadata_storage = DuckDBStorage(config["storage"]["duckdb"]["database"])
    graph_storage = GraphDBStorage(config["storage"]["graph_db"])
    
    try:
        # Step 1: Semantic search
        print("1. Performing semantic search...")
        search_results = vector_store.search(query, top_k=5)
        print(f"   Found {len(search_results)} relevant chunks")
        
        # Step 2: Retrieve full documents
        print("2. Retrieving full documents...")
        documents = []
        for result in search_results:
            doc = metadata_storage.get_document(result["doc_id"])
            if doc:
                documents.append({
                    "doc": doc,
                    "relevance": result["similarity"]
                })
        
        print(f"   Retrieved {len(documents)} documents")
        
        # Step 3: Find related documents
        print("3. Finding related documents...")
        related_docs = []
        for doc_data in documents[:3]:  # Top 3
            related = graph_storage.find_related_documents(
                doc_data["doc"]["doc_id"], 
                max_depth=1
            )
            related_docs.extend(related)
        
        print(f"   Found {len(related_docs)} related documents")
        
        # Step 4: Generate response context
        print("\n📄 Retrieved Context:")
        print("-" * 40)
        
        for i, doc_data in enumerate(documents[:3]):
            doc = doc_data["doc"]
            print(f"\nDocument {i+1}: {doc['file_name']}")
            print(f"Category: {doc.get('content_type', [])}")
            print(f"Relevance: {doc_data['relevance']:.3f}")
            print(f"Preview: {doc.get('text_content', '')[:150]}...")
        
        return {
            "query": query,
            "documents": documents,
            "related_documents": related_docs,
            "total_context_size": len(documents) + len(related_docs)
        }
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return None


def main():
    """Main test entry point."""
    
    # Configuration
    config = {
        "storage": {
            "duckdb": {
                "database": "/data/metadata.db"
            },
            "vector_store": {
                "host": "localhost",  # Change to "chroma" for Docker
                "port": 8000
            },
            "graph_db": {
                "uri": "bolt://localhost:7687",  # Change to "neo4j:7687" for Docker
                "user": "neo4j",
                "password": "password",
                "database": "neo4j"
            }
        }
    }
    
    # Update paths for Docker if needed
    if os.environ.get("DOCKER_ENV"):
        config["storage"]["vector_store"]["host"] = "chroma"
        config["storage"]["graph_db"]["uri"] = "bolt://neo4j:7687"
    
    # Run tests
    tester = RAGPipelineTester(config)
    tester.run_all_tests()
    
    # Example queries
    print("\n" + "="*60)
    print("EXAMPLE RAG QUERIES")
    print("="*60)
    
    example_queries = [
        "What spreadsheet data do we have?",
        "Show me information about presentations",
        "Find emails about meetings",
        "What research documents are available?"
    ]
    
    for query in example_queries:
        query_rag_system(query, config)
        time.sleep(1)  # Avoid overwhelming the system


if __name__ == "__main__":
    main()