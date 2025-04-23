import logging
import sys
from pathlib import Path
from typing import Optional
from .config import LoggingConfig

class Logger:
    """Advanced logging utility with file and console handlers."""
    
    def __init__(self, name: str, config: Optional[LoggingConfig] = None):
        """Initialize logger with given name and configuration."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatters
        formatter = logging.Formatter(
            config.format if config else LoggingConfig().format,
            datefmt=config.date_format if config else LoggingConfig().date_format
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(
            getattr(logging, config.console_level if config else LoggingConfig().console_level)
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if config and config.log_file:
            log_file = Path(config.log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(
                getattr(logging, config.file_level if config else LoggingConfig().file_level)
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(msg, *args, **kwargs)
    
    def performance(self, msg: str, *args, **kwargs):
        """Log performance-related message."""
        self.logger.info(f"PERF: {msg}", *args, **kwargs)
    
    def benchmark(self, msg: str, *args, **kwargs):
        """Log benchmark-related message."""
        self.logger.info(f"BENCH: {msg}", *args, **kwargs)
    
    def system(self, msg: str, *args, **kwargs):
        """Log system-related message."""
        self.logger.info(f"SYS: {msg}", *args, **kwargs)

def setup_logging(config: Optional[LoggingConfig] = None) -> Logger:
    """Setup and return the main application logger."""
    return Logger("llm_turbo_bench", config) 