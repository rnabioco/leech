"""
Logging configuration for the leech package.

Provides structured logging with consistent formatting across all modules.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Configure structured logging for leech.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional path to log file. If provided, logs to both file and console
        format_string: Optional custom format string. If None, uses default format

    Returns:
        Configured logger for the 'leech' package

    Example:
        >>> from leech.logging_config import setup_logging
        >>> import logging
        >>> logger = setup_logging(level=logging.DEBUG, log_file=Path("leech.log"))
        >>> logger.info("Starting training")
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Create handlers
    handlers: list[logging.Handler] = []

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger for 'leech' package
    logger = logging.getLogger("leech")
    logger.setLevel(level)
    logger.handlers.clear()  # Remove any existing handlers
    for handler in handlers:
        logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance for the module

    Example:
        >>> from leech.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing read")
    """
    return logging.getLogger(f"leech.{name}")
