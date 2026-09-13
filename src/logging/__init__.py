"""Logging package: application logging setup.

Note: this package intentionally lives at ``src.logging`` (per project
structure). Inside it, always import the stdlib module explicitly where
needed; the public API here is :func:`setup_logging` and :func:`get_logger`.
"""

from src.logging.logger import get_logger, set_level, setup_logging

__all__ = ["get_logger", "set_level", "setup_logging"]
