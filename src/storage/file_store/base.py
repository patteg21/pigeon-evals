from abc import ABC, abstractmethod

from src.utils.types import DocumentChunk, SECDocument


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
    def export_documents(self, data: SECDocument = None) -> bool:
        """Save files based on configuration"""
        
        raise NotImplementedError
    