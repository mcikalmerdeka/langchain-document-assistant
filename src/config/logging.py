"""
Centralized logging configuration for the RAG application.
"""
import logging
import sys
from pathlib import Path
from .settings import CHROMA_PERSIST_DIR

# Default log file location
DEFAULT_LOG_FILE = Path(CHROMA_PERSIST_DIR).parent / 'logs' / 'app.log'


def setup_logger(name: str = "docuschat", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup centralized logging for the application.
    
    Args:
        name: Logger name (default: "docuschat")
        log_file: Optional log file path. If None, only console logging is enabled.
                 Defaults to DEFAULT_LOG_FILE if not specified.
        level: Logging level (default: logging.INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not False:  # False means explicitly disable file logging
        log_path = Path(log_file) if log_file else DEFAULT_LOG_FILE
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name. If None, returns the root 'docuschat' logger.
    
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"docuschat.{name}")
    return logging.getLogger("docuschat")
