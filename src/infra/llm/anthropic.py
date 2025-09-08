from .base import LLMBaseClient
from typing import Dict, Any, Optional
from utils.logger import logger
import anthropic
import os

class AnthropicLLM(LLMBaseClient):
    """Anthropic LLM client."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "claude-3-haiku-20240307")
        self.api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found in config or environment variables")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        logger.info(f"Initializing Anthropic LLM with model: {self.model}")
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    def invoke(self, prompt: Optional[str] = None, query: Optional[str] = None, **kwargs) -> str:
        """Invoke Anthropic with a prompt or query."""
        if prompt and query:
            raise ValueError("Cannot specify both prompt and query")
        if not prompt and not query:
            raise ValueError("Must specify either prompt or query")
        
        text = prompt or query
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                messages=[
                    {"role": "user", "content": text}
                ]
            )
            
            result = response.content[0].text
            logger.info(f"Anthropic invoke successful, response length: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {str(e)}")
            raise