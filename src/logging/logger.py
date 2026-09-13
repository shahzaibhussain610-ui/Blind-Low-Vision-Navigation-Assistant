"""Application logger: file + console logging driven by config/logging.yaml."""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

import yaml

from src.utils.paths import LOGS_DIR, LOGGING_CONFIG_FILE

_CONFIG: Optional[dict] = None
_INITIALIZED = False

#: Valid log levels accepted by setup_logging.
VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _load_logging_config(config_file: Path) -> dict:
    """Read the logging YAML file, returning an empty dict if missing/invalid."""
    try:
        with open(config_file, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data.get("logging", {}) if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        # Logging must never crash the application; fall back to defaults.
        return {}


def setup_logging(
    config_file: Path = LOGGING_CONFIG_FILE,
    level_override: Optional[str] = None,
) -> logging.Logger:
    """Initialize the root application logger.

    Installs a rotating file handler under ``logs/`` and (optionally) a
    console handler. Safe to call multiple times; handlers are configured
    only once per process.

    Args:
        config_file: Path to ``config/logging.yaml``.
        level_override: Optional level (e.g. ``"DEBUG"``) that overrides
            both config and environment for the session.

    Returns:
        The configured root application logger named ``"app"``.
    """
    global _CONFIG, _INITIALIZED

    if _INITIALIZED:
        return logging.getLogger("app")

    _CONFIG = _load_logging_config(config_file)

    file_cfg: dict = _CONFIG.get("file", {})
    console_cfg: dict = _CONFIG.get("console", {})
    fmt: str = _CONFIG.get(
        "format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    date_fmt: str = _CONFIG.get("date_format", "%Y-%m-%d %H:%M:%S")

    level_name = (
        level_override or os.getenv("LOG_LEVEL") or _CONFIG.get("level", "INFO")
    ).upper()
    if level_name not in VALID_LEVELS:
        level_name = "INFO"
    level = getattr(logging, level_name)

    log_dir = LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / file_cfg.get("name", "application.log")

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    root = logging.getLogger("app")
    root.setLevel(level)
    # Do not propagate to the stdlib root logger to avoid duplicate lines.
    root.propagate = False

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if console_cfg.get("enabled", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    _INITIALIZED = True
    root.info("Logging initialized (level=%s, file=%s)", level_name, log_file)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``app`` namespace.

    Ensures logging is initialized even if a module logs before startup.
    """
    if not _INITIALIZED:
        setup_logging()
    return logging.getLogger(f"app.{name}")


def set_level(level: str) -> bool:
    """Change the active log level at runtime. Returns True on success."""
    level_name = (level or "").upper()
    if level_name not in VALID_LEVELS:
        return False
    logging.getLogger("app").setLevel(getattr(logging, level_name))
    return True
