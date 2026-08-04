"""
Banking RAG Centralized Logger module.

Configures structured logging for console and file output with configurable levels and formatting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "banking_rag",
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    console_level: Optional[str] = None,
) -> logging.Logger:
    """Configures and returns a logger instance.

    Args:
        name: Name of the logger.
        log_level: Logging level string used by the file handler (e.g. 'DEBUG', 'INFO',
            'WARNING', 'ERROR'). Also used by the console handler if console_level is omitted.
        log_file: Path to a file where log entries should be saved.
        console_level: Optional separate, typically quieter, level for console output only
            (e.g. 'WARNING' to hide routine INFO pipeline logs from the terminal while the
            file handler still records everything at log_level). Defaults to log_level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    console_numeric_level = getattr(logging, (console_level or log_level).upper(), numeric_level)

    # Logger itself must allow the more verbose of the two levels through, or the quieter
    # handler would never even get a chance to filter it.
    logger.setLevel(min(numeric_level, console_numeric_level))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Convenience getter for module-specific loggers.

    Args:
        module_name: Module or class name (typically __name__).

    Returns:
        logging.Logger instance prefixed with system root namespace.
    """
    return logging.getLogger(f"banking_rag.{module_name}")
