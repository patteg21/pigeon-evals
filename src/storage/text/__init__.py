from .base import TextStorageBase, TextStorageError
from .sqlite import SQLiteDB
from utils.logger import logger
from pathlib import Path


class TextStorage:
    """Auto-configuring text storage that reads from YAML config."""
    
    def __new__(cls) -> TextStorageBase:
        """Create text storage instance by auto-discovering config."""
        # Look for config file automatically
        config_paths = ["configs/test.yml", "config.yml", "test.yml"]
        for config_path in config_paths:
            if Path(config_path).exists():
                logger.info(f"Auto-loading text storage config from {config_path}")
                from src.utils.types.configs import YamlConfig
                yaml_config = YamlConfig.from_yaml(config_path)
                if yaml_config.storage and yaml_config.storage.text_store:
                    config_dict = yaml_config.storage.text_store.model_dump()
                    provider = config_dict.get("client", "sqlite")
                    
                    if provider == "sqlite":
                        logger.info("Creating SQLite text storage")
                        return SQLiteDB(config_dict)
                    else:
                        logger.warning(f"Unknown text provider '{provider}', falling back to SQLite")
                        return SQLiteDB(config_dict)
                break
        
        # Final fallback - use default
        logger.info("No text storage config found, using default SQLite")
        return SQLiteDB({})


__all__ = ["TextStorageBase", "SQLiteDB", "TextStorage"]