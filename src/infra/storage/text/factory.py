from .base import TextStorageBase
from .sqlite import SQLiteDB
from utils.logger import logger
from pathlib import Path
from typing import Dict, Any, Optional


class TextStorageFactory:
    """Factory for creating text storage instances based on provider."""
    
    _providers = {
        "sqlite": SQLiteDB,
        # "postgres": PostgresDB,    # Add when implemented
        # "s3": S3Storage,          # Add when implemented
        # "file": FileStore,        # Add when implemented
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
    def create_from_config(cls, config_path: Optional[str] = None) -> TextStorageBase:
        """Create text storage instance by auto-discovering or using provided config."""
        if config_path:
            config_paths = [config_path]
        else:
            config_paths = ["configs/test.yml", "config.yml", "test.yml"]
        
        for path in config_paths:
            if Path(path).exists():
                logger.info(f"Auto-loading text storage config from {path}")
                from src.utils.types.configs import YamlConfig
                yaml_config = YamlConfig.from_yaml(path)
                if yaml_config.storage and yaml_config.storage.text_store:
                    config_dict = yaml_config.storage.text_store.model_dump()
                    provider = config_dict.get("client", "sqlite")
                    return cls.create(provider, config_dict)
                break
        
        logger.info("No text storage config found, using default SQLite")
        return cls.create("sqlite", {})