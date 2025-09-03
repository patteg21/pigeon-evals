from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.utils.types.chunks import DocumentChunk

class RerankerError(Exception):
    """Base exception for Reranker operations"""
    pass


class RerankerBase(ABC):
    """Abstract base class for vector storage implementations"""
    
    @abstractmethod
    def rerank(
            self, 
            documents: List[Dict[str, Any]], 
            query: str
        ) -> List[Dict[str, Any]]:
        """Upload a DocumentChunk with embeddings to the vector database"""
        raise NotImplementedError    