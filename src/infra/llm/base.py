from abc import ABC, abstractmethod
from typing import Optional
from models.configs.eval import EvaluationConfig

class LLMBaseClient(ABC):

    def __init__(self, config: EvaluationConfig):
        super().__init__()
        self.config = config
        self.model = self.config.model
    
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