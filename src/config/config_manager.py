"""Configuration manager: loads and validates YAML configuration."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration cannot be loaded or is invalid."""


class ConfigManager:
    """Load, validate, and expose nested YAML configuration values.

    Access values with dot-separated keys::

        config.get("application.name")
        config.get("performance.target_fps", default=30)
    """

    #: Sections that must exist in app.yaml for the application to start.
    REQUIRED_SECTIONS: tuple = (
        "application",
        "system",
        "performance",
        "logging",
        "paths",
    )

    def __init__(self, config_file: Path) -> None:
        self._config_file = Path(config_file)
        self._config: dict = {}
        self.load()

    # ------------------------------------------------------------------ #
    # Loading / validation
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Load and validate the YAML configuration file.

        Raises:
            ConfigurationError: If the file is missing, unreadable,
                syntactically invalid, or missing required sections.
        """
        if not self._config_file.is_file():
            raise ConfigurationError(
                f"Configuration file could not be found: {self._config_file}\n"
                f"Please verify that '{self._config_file.name}' exists."
            )

        try:
            with open(self._config_file, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Configuration file is not valid YAML: {self._config_file}\n"
                f"Please check the file syntax. Details: {exc}"
            ) from exc
        except OSError as exc:
            raise ConfigurationError(
                f"Configuration file could not be read: {self._config_file}\n"
                f"Details: {exc}"
            ) from exc

        if not isinstance(loaded, dict):
            raise ConfigurationError(
                f"Configuration root must be a mapping (dict), "
                f"got: {type(loaded).__name__} in {self._config_file}"
            )

        self._config = loaded
        env_file = self._config_file.parent.parent / ".env"
        load_dotenv(env_file, override=False)
        if os.getenv("APP_ENV"):
            self._config.setdefault("application", {})["environment"] = os.environ["APP_ENV"]
        if os.getenv("LOG_LEVEL"):
            self._config.setdefault("logging", {})["level"] = os.environ["LOG_LEVEL"].upper()
        self._validate()
        logger.debug("Configuration loaded from %s", self._config_file)

    def _validate(self) -> None:
        """Ensure required top-level sections exist."""
        missing = [s for s in self.REQUIRED_SECTIONS if s not in self._config]
        if missing:
            raise ConfigurationError(
                f"Configuration is missing required section(s): "
                f"{', '.join(missing)} in {self._config_file}"
            )

    def reload(self) -> None:
        """Re-read the configuration from disk."""
        self.load()

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Return a configuration value using a dot-separated key path.

        Args:
            key: Dot path such as ``"application.name"``.
            default: Value returned when the key does not exist.

        Returns:
            The configuration value, or ``default`` if not found.
        """
        node: Any = self._config
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def has(self, key: str) -> bool:
        """Return True when the given dot-path key exists."""
        sentinel = object()
        return self.get(key, default=sentinel) is not sentinel

    @property
    def config_file(self) -> Path:
        """Path of the YAML file backing this manager."""
        return self._config_file

    def as_dict(self) -> dict:
        """Return a deep copy of the full configuration dictionary."""
        return copy.deepcopy(self._config)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"ConfigManager(file={self._config_file!s})"
