from abc import ABC, abstractmethod
from src.models.document import Document

class BaseExtractor(ABC):
    """Base class for document text extractors."""
    
    @abstractmethod
    def extract(self, file_path: str) -> Document:
        """Extract text and metadata from a document."""
        pass
