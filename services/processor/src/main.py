# src/main.py
import hydra
from omegaconf import DictConfig
import os
import sys
from typing import List

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.pdf_processor import PDFProcessor
from src.pipeline.chunkers.text_chunker import ContentAwareChunker
from src.pipeline.classifier import DocumentClassifier
from src.storage.duckdb import DuckDBStorage
from src.storage.vector_store import VectorStore  # You'll need to implement this
from src.storage.graph_db import GraphDBStorage  # You'll need to implement this

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for the document processing pipeline."""
    print(f"Running with config: {cfg}")
    
    # Initialize pipeline components
    pdf_processor = PDFProcessor(cfg.processor)
    chunker = ContentAwareChunker(cfg.chunker)
    classifier = DocumentClassifier(cfg.classifier)
    
    # Initialize storage
    metadata_storage = DuckDBStorage(cfg.storage.duckdb_path)
    vector_store = VectorStore(cfg.storage.vector_store_path)
    graph_storage = GraphDBStorage(cfg.storage.graph_db_url)
    
    # Process documents from input directory
    input_dir = cfg.processing.input_dir
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return
    
    # Find all PDF files
    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    for file_path in pdf_files:
        try:
            print(f"Processing: {file_path}")
            
            # Extract text and metadata with content type detection
            document = pdf_processor.process_pdf(file_path)
            
            # Classify document (add tags and persuasion strategies)
            document = classifier.classify(document)
            
            # Chunk document (content-aware)
            document = chunker.chunk_document(document)
            
            # Store document metadata
            metadata_storage.store_document(document)
            
            # Store chunks in vector store
            for chunk in document.chunks:
                vector_store.store_chunk(chunk)
            
            # Store document graph relationships
            graph_storage.store_document_relations(document)
            
            print(f"Successfully processed: {file_path}")
            print(f"  Content Types: {document.content_type}")
            print(f"  Chunks: {len(document.chunks)}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("Processing complete")

if __name__ == "__main__":
    main()