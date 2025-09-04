from abc import ABC, abstractmethod

from utils.types import DocumentChunk, Document


class FileStorageError(Exception):
    """Base exception for file storage operations"""
    pass


class FileStorageBase(ABC):
    """Abstract base class for file storage implementations"""
    
    @abstractmethod
    def export_chunks(self, data: DocumentChunk = None) -> bool:
        """Save files based on configuration"""
        raise NotImplementedError

    @abstractmethod
    def export_documents(self, data: Document = None) -> bool:
        """Save files based on configuration"""
        
        raise NotImplementedError
    