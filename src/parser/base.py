from abc import ABC, abstractmethod
from typing import Any, Dict

from models.configs.config import ParserConfig


class BaseParser(ABC):
    """Base class for all document processors."""
    
    def __init__(self, config: ParserConfig | None):
        self.config = config
    
    @abstractmethod
    def process(self):
        """Process a single document and return chunks."""
        raise NotImplementedError
    
    def process_batch(self):
        """Process a batch of documents and return all chunks."""
        pass
    
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
    