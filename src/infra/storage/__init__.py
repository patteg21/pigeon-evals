from .vector import VectorStorage, VectorStorageBase
from .text import TextStorage, TextStorageBase
from utils.logger import logger
from pathlib import Path


class VectorDB:
    """Auto-configuring vector database that reads from YAML config."""
    
    def __new__(cls) -> VectorStorageBase:
        """Create vector DB instance by auto-discovering config."""
        # Look for config file automatically
        config_paths = ["configs/test.yml", "config.yml", "test.yml"]
        for config_path in config_paths:
            if Path(config_path).exists():
                logger.info(f"Auto-loading vector DB config from {config_path}")
                from src.utils.types.configs import YamlConfig
                yaml_config = YamlConfig.from_yaml(config_path)
                if yaml_config.storage and yaml_config.storage.vector:
                    config_dict = yaml_config.storage.vector.model_dump()
                    provider = config_dict.get("provider", "faiss")
                    
                    from .vector.faiss import FAISSVectorDB
                    if provider == "faiss":
                        logger.info("Creating FAISS vector database")
                        return FAISSVectorDB(config_dict)
                    else:
                        logger.warning(f"Unknown vector provider '{provider}', falling back to FAISS")
                        return FAISSVectorDB(config_dict)
                break
        
        # Final fallback - use default
        logger.info("No vector DB config found, using default FAISS")
        from .vector.faiss import FAISSVectorDB
        return FAISSVectorDB({})


class TextDB:
    """Auto-configuring text database that reads from YAML config."""
    
    def __new__(cls) -> TextStorageBase:
        """Create text DB instance by auto-discovering config."""
        # Look for config file automatically
        config_paths = ["configs/test.yml", "config.yml", "test.yml"]
        for config_path in config_paths:
            if Path(config_path).exists():
                logger.info(f"Auto-loading text DB config from {config_path}")
                from src.utils.types.configs import YamlConfig
                yaml_config = YamlConfig.from_yaml(config_path)
                if yaml_config.storage and yaml_config.storage.text_store:
                    config_dict = yaml_config.storage.text_store.model_dump()
                    provider = config_dict.get("client", "sqlite")
                    
                    from .text.sqlite import SQLiteDB
                    if provider == "sqlite":
                        logger.info("Creating SQLite text database")
                        return SQLiteDB(config_dict)
                    else:
                        logger.warning(f"Unknown text provider '{provider}', falling back to SQLite")
                        return SQLiteDB(config_dict)
                break
        
        # Final fallback - use default
        logger.info("No text DB config found, using default SQLite")
        from .text.sqlite import SQLiteDB
        return SQLiteDB({})


__all__ = ["VectorDB", "TextDB", "VectorStorageBase", "TextStorageBase"]