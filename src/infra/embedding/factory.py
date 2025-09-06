from .base import BaseEmbedder
from .openai_embedder import OpenAIEmbedder
from .huggingface_embedder import HuggingFaceEmbedder
from utils.logger import logger
from pathlib import Path
from typing import Dict, Any, Optional


class EmbedderFactory:
    """Factory for creating embedder instances based on provider."""
    
    _providers = {
        "huggingface": HuggingFaceEmbedder,
        "openai": OpenAIEmbedder,
    }
    
    @classmethod
    def create(cls, provider: str, config: Dict[str, Any]) -> BaseEmbedder:
        """Create an embedder instance for the specified provider."""
        if provider not in cls._providers:
            logger.warning(f"Unknown embedder provider '{provider}', falling back to HuggingFace")
            provider = "huggingface"
        
        embedder_class = cls._providers[provider]
        logger.info(f"Creating {provider.title()} embedder with model: {config.get('model', 'default')}")
        return embedder_class(config)
    
    @classmethod
    def create_from_config(cls, config_path: Optional[str] = None) -> BaseEmbedder:
        """Create embedder instance by auto-discovering or using provided config."""
        if config_path:
            config_paths = [config_path]
        else:
            config_paths = ["configs/test.yml", "config.yml", "test.yml"]
        
        for path in config_paths:
            if Path(path).exists():
                logger.info(f"Auto-loading embedder config from {path}")
                from src.utils.types.configs import YamlConfig
                yaml_config = YamlConfig.from_yaml(path)
                if yaml_config.embedding:
                    config_dict = yaml_config.embedding.model_dump()
                    provider = config_dict.get("provider", "huggingface")
                    return cls.create(provider, config_dict)
                break
        
        logger.info("No embedding config found, using default HuggingFace embedder")
        return cls.create("huggingface", {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "pooling_strategy": "mean",
            "use_threading": True
        })