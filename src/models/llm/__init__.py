from .base import LLMBaseClient
from .openai import OpenAILLM
from .anthropic import AnthropicLLM
from .gemini import GeminiLLM
from utils.logger import logger
from pathlib import Path


class LLM:
    """Auto-configuring LLM that reads from YAML config."""
    
    def __new__(cls) -> LLMBaseClient:
        """Create LLM instance by auto-discovering config."""
        # Look for config file automatically
        config_paths = ["configs/test.yml", "config.yml", "test.yml"]
        for config_path in config_paths:
            if Path(config_path).exists():
                logger.info(f"Auto-loading LLM config from {config_path}")
                from src.utils.types.configs import YamlConfig
                yaml_config = YamlConfig.from_yaml(config_path)
                if yaml_config.eval:
                    config_dict = yaml_config.eval.model_dump()
                    provider = config_dict.get("provider", "openai")
                    
                    if provider == "openai":
                        logger.info(f"Creating OpenAI LLM with model: {config_dict.get('model', 'default')}")
                        return OpenAILLM(config_dict)
                    elif provider == "anthropic":
                        logger.info(f"Creating Anthropic LLM with model: {config_dict.get('model', 'default')}")
                        return AnthropicLLM(config_dict)
                    elif provider == "gemini":
                        logger.info(f"Creating Gemini LLM with model: {config_dict.get('model', 'default')}")
                        return GeminiLLM(config_dict)
                    else:
                        logger.warning(f"Unknown LLM provider '{provider}', falling back to OpenAI")
                        return OpenAILLM(config_dict)
                break
        
        # Final fallback - use default
        logger.info("No LLM config found, using default OpenAI LLM")
        return OpenAILLM({
            "model": "gpt-4o"
        })


__all__ = ["LLMBaseClient", "OpenAILLM", "AnthropicLLM", "GeminiLLM", "LLM"]