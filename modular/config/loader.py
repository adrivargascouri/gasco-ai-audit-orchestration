"""Configuration loader for YAML and environment-based configuration."""
from pathlib import Path
from typing import Optional
import yaml
from .config_schema import AppConfig


class ConfigLoader:
    """Loads and manages application configuration."""

    @staticmethod
    def load_from_yaml(config_path: str | Path) -> AppConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            AppConfig: Loaded configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is invalid or schema doesn't match
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

        return AppConfig(**config_data)

    @staticmethod
    def load_default() -> AppConfig:
        """Load default configuration.

        Returns:
            AppConfig: Default configuration with all hardcoded values
        """
        return AppConfig()

    @staticmethod
    def load_or_default(config_path: Optional[str | Path] = None) -> AppConfig:
        """Load configuration from file if provided, otherwise use defaults.

        Args:
            config_path: Optional path to YAML configuration file

        Returns:
            AppConfig: Loaded or default configuration
        """
        if config_path is None:
            return ConfigLoader.load_default()
        return ConfigLoader.load_from_yaml(config_path)
