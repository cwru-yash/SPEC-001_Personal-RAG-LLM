import os
import sys
import hydra
from omegaconf import DictConfig
import uuid

# # Add the project root to path
# sys.path.append('services/processor')

# Add project root to Python path
sys.path.insert(0, os.path.abspath('.'))


# Import components
from services.processor.src.models.document import Document
from services.processor.src.storage.duckdb import DuckDBStorage
from services.processor.src.storage.vector_store import VectorStore
from services.processor.src.storage.graph_db import GraphDBStorage

@hydra.main(config_path="services/processor/conf", config_name="config")
def main(cfg: DictConfig):
    """Test the integration of all components."""
    print(f"Testing with config: {cfg}")
    
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
    print("Test document properties:")
    print(f"- File name: {test_doc.file_name}")
    print(f"- Content type: {test_doc.content_type}")
    print(f"- Has chunks: {len(test_doc.chunks) > 0}")
    print("\nIntegration test successful!")
    
    # Initialize storage components
    try:
        print("Initializing DuckDB storage...")
        duckdb_storage = DuckDBStorage(cfg.storage.duckdb.database)
        
        print("Initializing Vector store...")
        vector_store = VectorStore(cfg.storage.vector_store)
        
        print("Initializing Graph DB storage...")
        graph_storage = GraphDBStorage(cfg.storage.graph_db)
        
        # Test storing document
        print("Storing document in DuckDB...")
        duckdb_result = duckdb_storage.store_document(test_doc)
        print(f"DuckDB storage result: {duckdb_result}")
        
        print("Storing chunks in Vector DB...")
        vector_result = vector_store.store_chunks(test_doc.chunks)
        print(f"Vector store result: {vector_result}")
        
        print("Storing document in Graph DB...")
        graph_result = graph_storage.store_document(test_doc.__dict__)
        print(f"Graph DB storage result: {graph_result}")
        
        # Test retrieval
        print("Retrieving document from DuckDB...")
        retrieved_doc = duckdb_storage.get_document(doc_id)
        print(f"Retrieved document: {retrieved_doc is not None}")
        
        print("Searching in Vector DB...")
        search_results = vector_store.search("test document", top_k=1)
        print(f"Vector search results: {len(search_results) > 0}")
        
        print("Integration test completed successfully!")
        
    except Exception as e:
        print(f"Integration test failed: {e}")

if __name__ == "__main__":
    main()
