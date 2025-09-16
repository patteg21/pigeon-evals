from abc import ABC, abstractmethod
from typing import Optional, List

from models import DocumentChunk
from models.configs.storage import TextStoreConfig

class TextStorageError(Exception):
    """Base exception for text storage operations"""
    pass


class TextStorageBase(ABC):
    """Abstract base class for text storage implementations"""
    
    def __init__(self, config: TextStoreConfig):
        self.config = config
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
    def store_document(self, doc_id: str, doc_data: dict) -> bool:
        """Store document data in the text storage system"""
        pass
    
    def store_document_chunk(self, chunk: "DocumentChunk") -> bool:
        """Store DocumentChunk in the text storage system"""
        return self.store_document(chunk.id, {
            'text': chunk.text,
            'document_data': {
                'id': chunk.document.id,
                'name': chunk.document.name,
                'path': chunk.document.path,
                'text': chunk.document.text
            },
            'embedding': chunk.embedding
        })
    
    @abstractmethod
    def retrieve_document(self, doc_id: str) -> Optional[dict]:
        """Retrieve document by ID"""
        pass

    @abstractmethod
    def retrieve_documents(self, doc_ids: List[str]) -> List[dict]:
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