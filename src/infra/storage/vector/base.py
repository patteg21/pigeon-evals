from abc import ABC, abstractmethod
from typing import List, Optional, Any
from models import DocumentChunk
from models.configs.storage import VectorConfig

class VectorStorageError(Exception):
    """Base exception for vector storage operations"""
    pass


class VectorStorageBase(ABC):
    """Abstract base class for vector storage implementations"""
    
    def __init__(self, config: VectorConfig):
        self.config = config
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
    def upload(self, chunk: DocumentChunk) -> Any:
        """Upload a DocumentChunk with embeddings to the vector database"""
        pass
    
    @abstractmethod
    def retrieve_from_id(self, vector_id: str) -> Any:
        """Retrieve a vector by its ID"""
        pass
    
    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        include_metadata: bool = True,
        filter: Optional[dict] | None = None,
    ) -> Any:
        """Query the vector database for similar vectors"""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> Any:
        """Delete vectors by IDs"""
        pass
    
    @abstractmethod
    def clear(self) -> Any:
        """Clear all vectors from the database"""
        pass