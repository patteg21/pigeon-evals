from .base import LLMBaseClient
from .openai import OpenAILLM
from .anthropic import AnthropicLLM
from .gemini import GeminiLLM
from .bedrock import BedrockLLM
from utils.logger import logger
from pathlib import Path
from typing import Dict, Any, Optional


class LLMFactory:
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
    def create_from_config(cls, config_path: Optional[str] = None) -> LLMBaseClient:
        """Create LLM instance by auto-discovering or using provided config."""
        if config_path:
            config_paths = [config_path]
        else:
            config_paths = ["configs/test.yml", "config.yml", "test.yml"]
        
        for path in config_paths:
            if Path(path).exists():
                logger.info(f"Auto-loading LLM config from {path}")
                from src.utils.types.configs import YamlConfig
                yaml_config = YamlConfig.from_yaml(path)
                if yaml_config.eval:
                    config_dict = yaml_config.eval.model_dump()
                    provider = config_dict.get("provider", "openai")
                    return cls.create(provider, config_dict)
                break
        
        logger.info("No LLM config found, using default OpenAI LLM")
        return cls.create("openai", {"model": "gpt-4o"})