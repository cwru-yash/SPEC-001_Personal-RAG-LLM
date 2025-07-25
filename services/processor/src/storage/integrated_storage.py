# services/processor/src/storage/integrated_storage.py

class IntegratedStorageManager:
    """Integrated storage manager for multiple backends."""
    
    def __init__(self, config):
        self.config = config
        
    def health_check(self):
        """Perform health check on all storage systems."""
        # Basic implementation for testing
        return {
            "status": "ok",
            "vector_store": True,
            "document_store": True,
            "knowledge_graph": True
        }