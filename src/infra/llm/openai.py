from .base import LLMBaseClient
from typing import Optional
from utils.logger import logger
import openai
import os
from models.configs.eval import EvaluationConfig

class OpenAILLM(LLMBaseClient):
    """OpenAI LLM client."""

    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.model = self.config.model or "gpt-4o"
        self.api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found in config or environment variables")
        
        self.client = openai.OpenAI(api_key=self.api_key)
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
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": text}
                ],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000)
            )
            
            result = response.choices[0].message.content
            logger.info(f"OpenAI invoke successful, response length: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise