from .base import LLMBaseClient
from typing import Dict, Any, Optional
from utils.logger import logger

class OpenAILLM(LLMBaseClient):
    """OpenAI LLM client."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "gpt-3.5-turbo")
        logger.info(f"Initializing OpenAI LLM with model: {self.model}")
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def invoke(self, prompt: Optional[str] = None, query: Optional[str] = None, **kwargs) -> str:
        """Invoke OpenAI with a prompt or query."""
        if prompt and query:
            raise ValueError("Cannot specify both prompt and query")
        if not prompt and not query:
            raise ValueError("Must specify either prompt or query")
        
        text = prompt or query
        # TODO: Implement OpenAI API call
        logger.info(f"OpenAI invoke called with text length: {len(text)}")
        return f"OpenAI response to: {text[:50]}..."