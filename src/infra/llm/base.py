from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class LLMBaseClient(ABC):
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        self.model = self.config.get("model")
    
    @abstractmethod
    def invoke(self, prompt: Optional[str] = None, query: Optional[str] = None, **kwargs) -> str:
        """Invoke the LLM with either a prompt or query and return the response."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text (basic implementation, override for accuracy)."""
        return len(text.split())