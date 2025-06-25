"""Utility functions for the League of Legends champion tracker."""
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List

class LRUCache:
    """Simple LRU cache implementation."""
    
    def __init__(self, capacity: int = 100):
        """Initialize cache with given capacity."""
        self.capacity = capacity
        self.cache: Dict[str, Any] = {}
        self.usage_order: List[str] = []
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        # Update usage order
        self.usage_order.remove(key)
        self.usage_order.append(key)
        
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Add value to cache."""
        if key in self.cache:
            self.usage_order.remove(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used item
            oldest = self.usage_order.pop(0)
            self.cache.pop(oldest)
        
        self.cache[key] = value
        self.usage_order.append(key)

def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with given name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)
    file_handler = logging.FileHandler(f"logs/{name}.log")
    
    # Create file handler
    file_handler = logging.FileHandler(f"logs/{name}.log")
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Set level and add handlers
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def timeit(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logging.getLogger("performance").debug(
            f"{func.__name__} took {elapsed:.4f} seconds"
        )
        return result
    return wrapper