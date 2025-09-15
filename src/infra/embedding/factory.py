from .base import BaseEmbedder
from .openai_embedder import OpenAIEmbedder
from .huggingface_embedder import HuggingFaceEmbedder
from models.shared.base_factory import BaseFactory
from utils.logger import logger
from typing import Dict, Any, Optional

from models.configs import EmbeddingConfig


class EmbedderFactory(BaseFactory):
    """Factory for creating embedder instances based on provider."""
    
    _providers = {
        "huggingface": HuggingFaceEmbedder,
        "openai": OpenAIEmbedder,
    }
    
    @classmethod
    def create(cls, provider: str, config: EmbeddingConfig) -> BaseEmbedder:
        """Create an embedder instance for the specified provider."""
        if provider not in cls._providers:
            logger.warning(f"Unknown embedder provider '{provider}', falling back to HuggingFace")
            provider = "huggingface"
        
        embedder_class = cls._providers[provider]
        logger.info(f"Creating {provider.title()} embedder with model: {config.model}")
        return embedder_class(config)
    
    @classmethod
    def get_config_key(cls) -> str:
        return "embedding"

    @classmethod
    def get_default_provider(cls) -> str:
        return "huggingface"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "pooling_strategy": "mean",
            "use_threading": True
        }

    @classmethod
    def _extract_config_from_yaml(cls, yaml_config) -> Optional[Any]:
        return yaml_config.embedding

    @classmethod
    def _extract_provider_from_config(cls, config_dict: Dict[str, Any]) -> str:
        return config_dict.get("provider", "huggingface")