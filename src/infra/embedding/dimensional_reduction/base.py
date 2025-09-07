from abc import ABC, abstractmethod
from typing import List, Any, Dict
import numpy as np


class BaseDimensionalReducer(ABC):
    """Base class for dimensional reduction techniques."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, embeddings: List[List[float]]) -> "BaseDimensionalReducer":
        """Fit the reducer on embeddings data."""
        pass
    
    @abstractmethod
    def transform(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Transform embeddings using fitted reducer."""
        pass
    
    def fit_transform(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Fit and transform embeddings in one step."""
        return self.fit(embeddings).transform(embeddings)
    
    @abstractmethod
    def save(self, path: str = None) -> None:
        """Save the fitted model."""
        pass
    
    @abstractmethod
    def load(self, path: str = None) -> "BaseDimensionalReducer":
        """Load a fitted model."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the reducer name."""
        pass