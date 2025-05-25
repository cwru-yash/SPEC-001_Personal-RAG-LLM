# test_enhanced_pipeline.py
import os
import sys
sys.path.append('.')

from services.processor.src.pipeline.zip_dataset_processor import ZipDatasetProcessor
from services.processor.src.pipeline.storage_integration import StorageIntegration

# Configuration
config = {
    "max_workers": 4,
    "pipeline": {
        "pdf": {"engine": "pymupdf", "extract_images": True, "perform_ocr": True},
        "office": {"excel": {"extract_all_sheets": True}},
        "image": {"ocr": {"engine": "tesseract"}},
        "chunker": {"default_chunk_size": 500}
    },
    "storage": {
        "duckdb": {"database": "/data/metadata.db"},
        "vector_store": {"host": "localhost", "port": 8000},
        "graph_db": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}
    }
}

# Process dataset
processor = ZipDatasetProcessor(config)
document_pairs = processor.process_dataset("/data/input", "/data/output")

# Store in databases
storage = StorageIntegration(config["storage"])
for pair in document_pairs:
    if pair.content_doc and pair.metadata_doc:
        storage.store_document_pair(pair.content_doc, pair.metadata_doc)

print(f"Processed {len(document_pairs)} pairs")
print(f"Storage stats: {storage.get_statistics()}")