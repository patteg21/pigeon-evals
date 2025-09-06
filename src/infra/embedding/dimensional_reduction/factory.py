from .base import BaseDimensionalReducer
from .pca_reducer import PCAReducer
from utils.logger import logger
from pathlib import Path
from typing import Dict, Any, Optional


class DimensionalReductionFactory:
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
    def create_from_config(cls, config_path: Optional[str] = None) -> Optional[BaseDimensionalReducer]:
        """Create dimensional reducer instance by auto-discovering or using provided config."""
        if config_path:
            config_paths = [config_path]
        else:
            config_paths = ["configs/test.yml", "config.yml", "test.yml"]
        
        for path in config_paths:
            if Path(path).exists():
                logger.info(f"Auto-loading dimensional reduction config from {path}")
                from src.utils.types.configs import YamlConfig
                yaml_config = YamlConfig.from_yaml(path)
                if yaml_config.embedding and hasattr(yaml_config.embedding, 'dimension_reduction'):
                    dr_config = yaml_config.embedding.dimension_reduction
                    if dr_config:
                        config_dict = dr_config.model_dump() if hasattr(dr_config, 'model_dump') else dr_config
                        reducer_type = config_dict.get("type", "pca")
                        return cls.create(reducer_type, config_dict)
                break
        
        logger.info("No dimensional reduction config found, skipping")
        return None