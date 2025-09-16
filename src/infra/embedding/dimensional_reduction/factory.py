from .base import BaseDimensionalReducer
from .pca_reducer import PCAReducer
from utils.logger import logger
from utils.config_manager import ConfigManager
from models.configs.embedding import DimensionReduction



class DimensionalReductionFactory:
    """Factory for creating dimensional reduction instances based on type."""

    _reducers = {
        "pca": PCAReducer,
        "PCA": PCAReducer,  # Support uppercase variant from config
    }

    @classmethod
    def create_reducer(cls, dimension_config: DimensionReduction) -> Optional[BaseDimensionalReducer]:
        """Create dimensional reducer instance from config object."""
        if not dimension_config:
            return None

        reducer_type = dimension_config.type

        if reducer_type not in cls._reducers:
            logger.warning(f"Unknown reducer type '{reducer_type}', falling back to PCA")
            reducer_type = "pca"

        reducer_class = cls._reducers[reducer_type]
        logger.info(f"Creating {reducer_type.upper()} dimensional reducer")
        return reducer_class(dimension_config)


    @classmethod
    def create_from_config(cls) -> Optional[BaseDimensionalReducer]:
        """Create dimensional reducer instance using the singleton config."""
        config_manager = ConfigManager()
        config = config_manager.config

        if config.embedding and config.embedding.dimension_reduction:
            return cls.create_reducer(config.embedding.dimension_reduction)

        else:
            logger.info("No dimensional reduction config found, skipping")
            return None