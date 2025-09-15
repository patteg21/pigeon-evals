from .base import LLMBaseClient
from .openai import OpenAILLM
from .anthropic import AnthropicLLM
from .gemini import GeminiLLM
from .bedrock import BedrockLLM
from models.shared.base_factory import BaseFactory
from utils.logger import logger
from typing import Dict, Any, Optional


class LLMFactory(BaseFactory):
    """Factory for creating LLM instances based on provider."""
    
    _providers = {
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
        "gemini": GeminiLLM,
        "bedrock": BedrockLLM,
    }
    
    @classmethod
    def create(cls, provider: str, config: Dict[str, Any]) -> LLMBaseClient:
        """Create an LLM instance for the specified provider."""
        if provider not in cls._providers:
            logger.warning(f"Unknown LLM provider '{provider}', falling back to OpenAI")
            provider = "openai"
        
        llm_class = cls._providers[provider]
        logger.info(f"Creating {provider.title()} LLM with model: {config.get('model', 'default')}")
        return llm_class(config)
    
    @classmethod
    def get_config_key(cls) -> str:
        return "eval"

    @classmethod
    def get_default_provider(cls) -> str:
        return "openai"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {"model": "gpt-4o"}

    @classmethod
    def _extract_config_from_yaml(cls, yaml_config) -> Optional[Any]:
        return yaml_config.eval

    @classmethod
    def _extract_provider_from_config(cls, config_dict: Dict[str, Any]) -> str:
        return config_dict.get("provider", "openai")