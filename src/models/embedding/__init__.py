from .base import BaseEmbedder
from .openai_embedder import OpenAIEmbedder
from .huggingface_embedder import HuggingFaceEmbedder
from utils.logger import logger
from pathlib import Path


class Embedder:
    """Auto-configuring embedder that reads from YAML config."""
    
    def __new__(cls) -> BaseEmbedder:
        """Create embedder instance by auto-discovering config."""
        # Look for config file automatically
        config_paths = ["configs/test.yml", "config.yml", "test.yml"]
        for config_path in config_paths:
            if Path(config_path).exists():
                logger.info(f"Auto-loading embedder config from {config_path}")
                from src.utils.types.configs import YamlConfig
                yaml_config = YamlConfig.from_yaml(config_path)
                if yaml_config.embedding:
                    config_dict = yaml_config.embedding.model_dump()
                    provider = config_dict.get("provider", "huggingface")
                    
                    if provider == "huggingface":
                        logger.info(f"Creating HuggingFace embedder with model: {config_dict.get('model', 'default')}")
                        return HuggingFaceEmbedder(config_dict)
                    elif provider == "openai":
                        logger.info(f"Creating OpenAI embedder with model: {config_dict.get('model', 'default')}")
                        return OpenAIEmbedder(config_dict)
                break
        
        # Final fallback - use default
        logger.info("No embedding config found, using default HuggingFace embedder")
        return HuggingFaceEmbedder({
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "pooling_strategy": "mean",
            "use_threading": True
        })


__all__ = ["BaseEmbedder", "OpenAIEmbedder", "HuggingFaceEmbedder", "Embedder"]