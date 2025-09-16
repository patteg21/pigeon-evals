from .base import BaseEmbedder
from .openai_embedder import OpenAIEmbedder
from .huggingface_embedder import HuggingFaceEmbedder
from utils.logger import logger
from utils.config_manager import ConfigManager
from models.configs import EmbeddingConfig


class EmbedderFactory:
    """Factory for creating embedder instances based on provider."""
    
    _providers = {
        "huggingface": HuggingFaceEmbedder,
        "openai": OpenAIEmbedder,
    }
    
    @classmethod
    def create_from_config(cls) -> BaseEmbedder:
        """Create an embedder instance using the singleton config."""
        config_manager = ConfigManager()
        config = config_manager.config

        if config.embedding:
            provider = config.embedding.provider
            if provider not in cls._providers:
                logger.warning(f"Unknown embedder provider '{provider}', falling back to HuggingFace")
                provider = "huggingface"

            embedder_class = cls._providers[provider]
            logger.info(f"Creating {provider.title()} embedder with model: {config.embedding.model}")
            return embedder_class(config.embedding)
        else:
            # Default fallback
            logger.info("No embedding config found, using default HuggingFace embedder")
            default_config = EmbeddingConfig(
                provider="huggingface",
                model="sentence-transformers/all-MiniLM-L6-v2",
                pooling_strategy="mean",
                use_threading=True
            )
            return cls._providers["huggingface"](default_config)