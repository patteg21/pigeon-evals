
from .base import VectorStorageBase, VectorStorageError
from .faiss import FAISSVectorDB
from utils.logger import logger
from pathlib import Path


class VectorStorage:
    """Auto-configuring vector storage that reads from YAML config."""
    
    def __new__(cls) -> VectorStorageBase:
        """Create vector storage instance by auto-discovering config."""
        # Look for config file automatically
        config_paths = ["configs/test.yml", "config.yml", "test.yml"]
        for config_path in config_paths:
            if Path(config_path).exists():
                logger.info(f"Auto-loading vector storage config from {config_path}")
                from src.utils.types.configs import YamlConfig
                yaml_config = YamlConfig.from_yaml(config_path)
                if yaml_config.storage and yaml_config.storage.vector:
                    config_dict = yaml_config.storage.vector.model_dump()
                    provider = config_dict.get("provider", "faiss")
                    
                    if provider == "faiss":
                        logger.info("Creating FAISS vector storage")
                        return FAISSVectorDB(config_dict)
                    else:
                        logger.warning(f"Unknown vector provider '{provider}', falling back to FAISS")
                        return FAISSVectorDB(config_dict)
                break
        
        # Final fallback - use default
        logger.info("No vector storage config found, using default FAISS")
        return FAISSVectorDB({})


__all__ = ["VectorStorageBase", "FAISSVectorDB", "VectorStorage"]