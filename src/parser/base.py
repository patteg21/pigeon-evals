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
        raise NotImplementedError
    
    def process_batch(self, documents: List[Document]) -> List[DocumentChunk]:
        """Process a batch of documents and return all chunks."""
        all_chunks = []
        for document in documents:
            chunks = self.process(document)
            all_chunks.extend(chunks)
        return all_chunks
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the processor name."""
        pass

    def _compute_page_number(self, body: str, pos: int) -> int:
        """Pages start at 1; count [PAGE_BREAK] before `pos`."""
        if pos < 0:
            pos = 0
        return body[:pos].count("[PAGE_BREAK]") + 1
    