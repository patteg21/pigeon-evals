from .base import LLMBaseClient
from typing import Dict, Any, Optional
from utils.logger import logger
import google.generativeai as genai
import os

class GeminiLLM(LLMBaseClient):
    """Google Gemini LLM client."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "gemini-pro")
        self.api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key not found in config or environment variables")
        
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)
        logger.info(f"Initializing Gemini LLM with model: {self.model}")
    
    @property
    def provider_name(self) -> str:
        return "gemini"
    
    def invoke(self, prompt: Optional[str] = None, query: Optional[str] = None, **kwargs) -> str:
        """Invoke Gemini with a prompt or query."""
        if prompt and query:
            raise ValueError("Cannot specify both prompt and query")
        if not prompt and not query:
            raise ValueError("Must specify either prompt or query")
        
        text = prompt or query
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", 0.7),
                max_output_tokens=kwargs.get("max_tokens", 1000)
            )
            
            response = self.client.generate_content(
                text,
                generation_config=generation_config
            )
            
            result = response.text
            logger.info(f"Gemini invoke successful, response length: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise