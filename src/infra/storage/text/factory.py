from .base import TextStorageBase
from .sqlite import SQLiteDB
from .postgres import PostgresDB
from .s3 import S3Storage
from .file_store import FileStore
from utils.logger import logger
from utils.config_manager import ConfigManager
from models.configs.storage import TextStoreConfig
from typing import Optional


class TextStorageFactory:
    """Factory for creating text storage instances based on provider."""
    
    _providers = {
        "sqlite": SQLiteDB,
        "postgres": PostgresDB,
        "s3": S3Storage,
        "file": FileStore,
    }
    
    @classmethod
    def create(cls, provider: str, config: TextStoreConfig) -> TextStorageBase:
        """Create a text storage instance for the specified provider."""
        if provider not in cls._providers:
            logger.warning(f"Unknown text provider '{provider}', falling back to SQLite")
            provider = "sqlite"

        storage_class = cls._providers[provider]
        logger.info(f"Creating {provider.upper()} text storage")
        return storage_class(config)

    @classmethod
    def create_from_config(cls) -> Optional[TextStorageBase]:
        """Create a text storage instance using the singleton config."""
        config_manager = ConfigManager()
        config = config_manager.config

        if config.storage and config.storage.text_store:
            text_config = config.storage.text_store
            provider = text_config.client

            if provider not in cls._providers:
                logger.warning(f"Unknown text provider '{provider}', falling back to SQLite")
                provider = "sqlite"

            storage_class = cls._providers[provider]
            logger.info(f"Creating {provider.upper()} text storage from config")
            return storage_class(text_config)
        else:
            logger.info("No text storage config found, skipping text storage")
            return None
    
