from .base import VectorStorageBase
from .faiss import FAISSVectorDB
from models.shared.base_factory import BaseFactory
from utils.logger import logger
from typing import Dict, Any, Optional


class VectorStorageFactory(BaseFactory):
    """Factory for creating vector storage instances based on provider."""
    
    _providers = {
        "faiss": FAISSVectorDB,
        # "pinecone": PineconeVectorDB,  # Add when implemented
        # "qdrant": QdrantVectorDB,      # Add when implemented
    }
    
    @classmethod
    def create(cls, provider: str, config: Dict[str, Any]) -> VectorStorageBase:
        """Create a vector storage instance for the specified provider."""
        if provider not in cls._providers:
            logger.warning(f"Unknown vector provider '{provider}', falling back to FAISS")
            provider = "faiss"
        
        storage_class = cls._providers[provider]
        logger.info(f"Creating {provider.upper()} vector storage")
        return storage_class(config)
    
    @classmethod
    def get_config_key(cls) -> str:
        return "vector storage"

    @classmethod
    def get_default_provider(cls) -> str:
        return "faiss"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def _extract_config_from_yaml(cls, yaml_config) -> Optional[Any]:
        return yaml_config.storage.vector if yaml_config.storage else None

    @classmethod
    def _extract_provider_from_config(cls, config_dict: Dict[str, Any]) -> str:
        return config_dict.get("provider", "faiss")