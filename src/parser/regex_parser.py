from abc import ABC, abstractmethod
from typing import Any, Dict, List
from utils.types import Document
from utils.types.chunks import DocumentChunk


class BaseProcessor(ABC):
    """Base class for all document processors."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @abstractmethod
    def process(self, document: Document) -> List[DocumentChunk]:
        """Process a single document and return chunks."""
        pass
    
    def process_batch(self, documents: List[Document]) -> List[DocumentChunk]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the processor name."""
        pass

    def _compute_page_number(self, body: str, pos: int) -> int:
        pass