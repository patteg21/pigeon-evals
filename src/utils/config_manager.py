from typing import Optional
from pathlib import Path
from models import YamlConfig


class ConfigManager:
    """Singleton configuration manager for the application."""

    _instance: Optional['ConfigManager'] = None
    _config: Optional[YamlConfig] = None

    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file {config_path} not found")

        self._config = YamlConfig.from_yaml(config_path)

    def get_config(self) -> YamlConfig:
        """Get the loaded configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config

    @property
    def config(self) -> YamlConfig:
        """Property access to configuration."""
        return self.get_config()