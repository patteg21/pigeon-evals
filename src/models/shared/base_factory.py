from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from utils.logger import logger


class BaseFactory(ABC):
    """Base factory class providing common config loading functionality."""

    @classmethod
    @abstractmethod
    def create(cls, provider: str, config: Dict[str, Any]) -> Any:
        """Create an instance for the specified provider."""
        pass

    @classmethod
    @abstractmethod
    def get_config_key(cls) -> str:
        """Return the YAML config key to look for (e.g., 'eval', 'embedding')."""
        pass

    @classmethod
    @abstractmethod
    def get_default_provider(cls) -> str:
        """Return the default provider to use when none is found."""
        pass

    @classmethod
    @abstractmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Return the default config to use when none is found."""
        pass

    @classmethod
    def get_config_paths(cls) -> List[str]:
        """Return the default config file paths to search."""
        return ["configs/test.yml", "config.yml", "test.yml"]

    @classmethod
    def create_from_config(cls, config_path: Optional[str] = None) -> Any:
        """Create instance by auto-discovering or using provided config."""
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

        logger.info(f"No {cls.get_config_key()} config found, using default {cls.get_default_provider()}")
        return cls.create(cls.get_default_provider(), cls.get_default_config())

    @classmethod
    @abstractmethod
    def _extract_config_from_yaml(cls, yaml_config) -> Optional[Any]:
        """Extract the relevant config object from the loaded YAML."""
        pass

    @classmethod
    @abstractmethod
    def _extract_provider_from_config(cls, config_dict: Dict[str, Any]) -> str:
        """Extract the provider name from the config dictionary."""
        pass