from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class TextStorageError(Exception):
    """Base exception for text storage operations"""
    pass


class TextStorageBase(ABC):
    """Abstract base class for text storage implementations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
    def store_document(self, doc_id: str, doc_data: Dict[str, Any]) -> bool:
        """Store document data in the text storage system"""
        pass
    
    @abstractmethod
    def retrieve_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID"""
        pass
    
    @abstractmethod
    def retrieve_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple documents by IDs"""
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get total number of documents"""
        pass
    
    @abstractmethod
    def clear_all(self) -> bool:
        """Clear all documents from storage"""
        pass