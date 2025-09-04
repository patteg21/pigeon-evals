from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from utils.types.chunks import DocumentChunk

class VectorStorageError(Exception):
    """Base exception for vector storage operations"""
    pass


class VectorStorageBase(ABC):
    """Abstract base class for vector storage implementations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
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
        filter: Optional[Dict[str, Any]] | None = None, 
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