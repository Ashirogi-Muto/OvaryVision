"""Logging utilities for OvaryVision enforcing rotating file handlers and standard formatting."""

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Optional

DEFAULT_LOG_DIR = Path(os.getenv("OVARYVISION_LOG_DIR", "logs"))
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
DEFAULT_BACKUP_COUNT = 5  # Maximum 5 rotating files
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "ovaryvision",
    log_dir: Optional[Path | str] = None,
    level: int = logging.INFO,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    log_to_console: bool = True,
) -> logging.Logger:
    """Configures and returns a rotating logger adhering to project rules.

    Args:
        name: Name of the logger.
        log_dir: Directory where log files should be stored. Defaults to /home/ubuntu/major/logs.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        max_bytes: Maximum file size before rotation (defaults to 10MB).
        backup_count: Maximum number of backup files (defaults to 5).
        log_to_console: Whether to attach a console StreamHandler.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DATE_FORMAT)

    # Resolve log directory
    resolved_log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    try:
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = resolved_log_dir / f"{name}.log"

        file_handler = RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as exc:
        # Fall back to console only if directory cannot be created/written
        logging.warning(
            "Could not initialize RotatingFileHandler at %s: %s. Continuing with console logging.",
            resolved_log_dir,
            exc,
        )

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Do not propagate to root logger to avoid duplicate log lines
    logger.propagate = False
    return logger


def get_logger(name: str = "ovaryvision") -> logging.Logger:
    """Retrieve an existing logger or create one with default settings."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name=name)
    return logger

