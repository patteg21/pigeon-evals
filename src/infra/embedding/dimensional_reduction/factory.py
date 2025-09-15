from .base import BaseDimensionalReducer
from .pca_reducer import PCAReducer
from models.shared.base_factory import BaseFactory
from utils.logger import logger
from pathlib import Path
from typing import Dict, Any, Optional


class DimensionalReductionFactory(BaseFactory):
    """Factory for creating dimensional reduction instances based on type."""
    
    _reducers = {
        "pca": PCAReducer,
        "PCA": PCAReducer,  # Support uppercase variant from config
    }
    
    @classmethod
    def create(cls, reducer_type: str, config: Dict[str, Any]) -> BaseDimensionalReducer:
        """Create a dimensional reducer instance for the specified type."""
        if reducer_type not in cls._reducers:
            logger.warning(f"Unknown reducer type '{reducer_type}', falling back to PCA")
            reducer_type = "pca"
        
        reducer_class = cls._reducers[reducer_type]
        logger.info(f"Creating {reducer_type.upper()} dimensional reducer with config: {config}")
        return reducer_class(config)
    
    @classmethod
    def get_config_key(cls) -> str:
        return "dimensional reduction"

    @classmethod
    def get_default_provider(cls) -> str:
        return "pca"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def _extract_config_from_yaml(cls, yaml_config) -> Optional[Any]:
        if yaml_config.embedding and hasattr(yaml_config.embedding, 'dimension_reduction'):
            return yaml_config.embedding.dimension_reduction
        return None

    @classmethod
    def _extract_provider_from_config(cls, config_dict: Dict[str, Any]) -> str:
        return config_dict.get("type", "pca")

    @classmethod
    def create_from_config(cls, config_path: Optional[str] = None) -> Optional[BaseDimensionalReducer]:
        """Create dimensional reducer instance by auto-discovering or using provided config."""
        if config_path:
            config_paths = [config_path]
        else:
            config_paths = cls.get_config_paths()

        for path in config_paths:
            if Path(path).exists():
                logger.info(f"Auto-loading {cls.get_config_key()} config from {path}")
                from models import YamlConfig
                yaml_config = YamlConfig.from_yaml(path)

                config_obj = cls._extract_config_from_yaml(yaml_config)
                if config_obj:
                    config_dict = config_obj.model_dump() if hasattr(config_obj, 'model_dump') else config_obj
                    provider = cls._extract_provider_from_config(config_dict)
                    return cls.create(provider, config_dict)
                break

        logger.info("No dimensional reduction config found, skipping")
        return None