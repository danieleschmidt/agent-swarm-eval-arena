"""Logging utilities for the Swarm Arena."""

import logging
import sys
from typing import Optional
from pathlib import Path
import json
from datetime import datetime


class SwarmArenaLogger:
    """Custom logger for Swarm Arena with structured logging."""
    
    def __init__(self, name: str = "swarm_arena", level: int = logging.INFO, 
                 log_file: Optional[str] = None) -> None:
        """Initialize the logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional structured data."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional structured data."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional structured data."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with optional structured data."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with optional structured data."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def _log_with_context(self, level: int, message: str, **kwargs) -> None:
        """Log message with additional context."""
        if kwargs:
            context = json.dumps(kwargs, separators=(',', ':'))
            full_message = f"{message} | Context: {context}"
        else:
            full_message = message
        
        self.logger.log(level, full_message)


# Global logger instance
logger = SwarmArenaLogger()


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> SwarmArenaLogger:
    """Setup global logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    global logger
    logger = SwarmArenaLogger("swarm_arena", level, log_file)
    return logger


def get_logger(name: str = "swarm_arena") -> SwarmArenaLogger:
    """Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return SwarmArenaLogger(name)