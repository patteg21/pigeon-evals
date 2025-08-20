from abc import ABC, abstractmethod
from typing import Any, List
from pathlib import Path


class FileStorageError(Exception):
    """Base exception for file storage operations"""
    pass


class FileStorageBase(ABC):
    """Abstract base class for file storage implementations"""
    
    @abstractmethod
    def save_files(self, config: Any, data: Any = None) -> bool:
        """Save files based on configuration"""
        pass
    
    @abstractmethod
    def load_files(self, config: Any) -> Any:
        """Load files based on configuration"""
        pass
    
    @abstractmethod
    def delete_files(self, config: Any) -> bool:
        """Delete files based on configuration"""
        pass
    
    @abstractmethod
    def list_files(self, config: Any) -> List[Path]:
        """List files based on configuration"""
        pass