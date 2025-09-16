from .base import VectorStorageBase
from .faiss import FAISSVectorDB
from models.shared.base_factory import BaseFactory
from utils.logger import logger
from utils.config_manager import ConfigManager
from models.configs.storage import VectorConfig
from typing import Optional



class VectorStorageFactory(BaseFactory):
    """Factory for creating vector storage instances based on provider."""
    
    _providers = {
        "faiss": FAISSVectorDB,
        # "pinecone": PineconeVectorDB,  # Add when implemented
        # "qdrant": QdrantVectorDB,      # Add when implemented
    }
    
    @classmethod
    def create(cls, provider: str, config: VectorConfig) -> VectorStorageBase:
        """Create a vector storage instance for the specified provider."""
        if provider not in cls._providers:
            logger.warning(f"Unknown vector provider '{provider}', falling back to FAISS")
            provider = "faiss"

        storage_class = cls._providers[provider]
        logger.info(f"Creating {provider.upper()} vector storage")
        return storage_class(config)

    @classmethod
    def create_from_config(cls) -> Optional[VectorStorageBase]:
        """Create a vector storage instance using the singleton config."""
        config_manager = ConfigManager()
        config = config_manager.config

        if config.storage and config.storage.vector:
            vector_config = config.storage.vector
            provider = vector_config.provider or "faiss"

            if provider not in cls._providers:
                logger.warning(f"Unknown vector provider '{provider}', falling back to FAISS")
                provider = "faiss"

            storage_class = cls._providers[provider]
            logger.info(f"Creating {provider.upper()} vector storage from config")
            return storage_class(vector_config)
        else:
            logger.info("No vector storage config found, skipping vector storage")
            return None

