from .base import VectorStorageBase
from .faiss import FAISSVectorDB
from utils.logger import logger
from pathlib import Path
from typing import Dict, Any, Optional


class VectorStorageFactory:
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
    def create_from_config(cls, config_path: Optional[str] = None) -> VectorStorageBase:
        """Create vector storage instance by auto-discovering or using provided config."""
        if config_path:
            config_paths = [config_path]
        else:
            config_paths = ["configs/test.yml", "config.yml", "test.yml"]
        
        for path in config_paths:
            if Path(path).exists():
                logger.info(f"Auto-loading vector storage config from {path}")
                from src.utils.types.configs import YamlConfig
                yaml_config = YamlConfig.from_yaml(path)
                if yaml_config.storage and yaml_config.storage.vector:
                    config_dict = yaml_config.storage.vector.model_dump()
                    provider = config_dict.get("provider", "faiss")
                    return cls.create(provider, config_dict)
                break
        
        logger.info("No vector storage config found, using default FAISS")
        return cls.create("faiss", {})