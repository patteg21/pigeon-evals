from .base import TextStorageBase
from .sqlite import SQLiteDB
from .postgres import PostgresDB
from .s3 import S3Storage
from .file_store import FileStore
from models.shared.base_factory import BaseFactory
from utils.logger import logger
from typing import Dict, Any, Optional


class TextStorageFactory(BaseFactory):
    """Factory for creating text storage instances based on provider."""
    
    _providers = {
        "sqlite": SQLiteDB,
        "postgres": PostgresDB,
        "s3": S3Storage,
        "file": FileStore,
    }
    
    @classmethod
    def create(cls, provider: str, config: Dict[str, Any]) -> TextStorageBase:
        """Create a text storage instance for the specified provider."""
        if provider not in cls._providers:
            logger.warning(f"Unknown text provider '{provider}', falling back to SQLite")
            provider = "sqlite"
        
        storage_class = cls._providers[provider]
        logger.info(f"Creating {provider.upper()} text storage")
        return storage_class(config)
    
    @classmethod
    def get_config_key(cls) -> str:
        return "text storage"

    @classmethod
    def get_default_provider(cls) -> str:
        return "sqlite"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def _extract_config_from_yaml(cls, yaml_config) -> Optional[Any]:
        return yaml_config.storage.text_store if yaml_config.storage else None

    @classmethod
    def _extract_provider_from_config(cls, config_dict: Dict[str, Any]) -> str:
        return config_dict.get("client", "sqlite")