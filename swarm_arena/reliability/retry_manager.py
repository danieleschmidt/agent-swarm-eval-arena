"""Retry management with exponential backoff and jitter."""

import time
import random
import functools
import logging
from typing import Callable, Any, Optional, List, Type, Union
from dataclasses import dataclass
from enum import Enum


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    exceptions: tuple = (Exception,)
    backoff_multiplier: float = 2.0
    name: Optional[str] = None


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Retry exhausted after {attempts} attempts. Last error: {last_exception}")


class RetryManager:
    """Manages retry logic with various strategies."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(f"retry_manager.{config.name or 'default'}")
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_attempts': 0,
            'max_attempts_used': 0
        }
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RetryExhaustedError: When all retry attempts fail
        """
        self.stats['total_calls'] += 1
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            self.stats['total_attempts'] += 1
            
            try:
                self.logger.debug(f"Attempt {attempt}/{self.config.max_attempts}")
                result = func(*args, **kwargs)
                
                self.stats['successful_calls'] += 1
                self.stats['max_attempts_used'] = max(
                    self.stats['max_attempts_used'], 
                    attempt
                )
                
                if attempt > 1:
                    self.logger.info(f"Function succeeded on attempt {attempt}")
                
                return result
                
            except self.config.exceptions as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt} failed: {e}")
                
                # Don't sleep after the last attempt
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    self.logger.debug(f"Retrying in {delay:.2f} seconds")
                    time.sleep(delay)
            
            except Exception as e:
                # Non-retryable exception
                self.stats['failed_calls'] += 1
                self.logger.error(f"Non-retryable exception: {e}")
                raise
        
        # All attempts exhausted
        self.stats['failed_calls'] += 1
        self.stats['max_attempts_used'] = max(
            self.stats['max_attempts_used'], 
            self.config.max_attempts
        )
        
        raise RetryExhaustedError(self.config.max_attempts, last_exception)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number.
        
        Args:
            attempt: Current attempt number (1-indexed)
            
        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
            
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
            
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.config.base_delay * self._fibonacci(attempt)
            
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return 1
        
        a, b = 1, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        
        return b
    
    def get_stats(self) -> dict:
        """Get retry statistics."""
        total_calls = self.stats['total_calls']
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_calls'] / total_calls 
                if total_calls > 0 else 0.0
            ),
            'average_attempts': (
                self.stats['total_attempts'] / total_calls 
                if total_calls > 0 else 0.0
            )
        }


def retry(max_attempts: int = 3,
          base_delay: float = 1.0,
          max_delay: float = 60.0,
          strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
          jitter: bool = True,
          exceptions: Union[Type[Exception], tuple] = Exception,
          backoff_multiplier: float = 2.0,
          name: Optional[str] = None) -> Callable:
    """Decorator to add retry logic to a function.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay between attempts
        max_delay: Maximum delay between attempts
        strategy: Retry strategy
        jitter: Whether to add random jitter
        exceptions: Exception types that trigger retry
        backoff_multiplier: Multiplier for exponential backoff
        name: Name for logging and stats
        
    Returns:
        Decorated function
    """
    if isinstance(exceptions, type):
        exceptions = (exceptions,)
    
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=strategy,
        jitter=jitter,
        exceptions=exceptions,
        backoff_multiplier=backoff_multiplier,
        name=name
    )
    
    manager = RetryManager(config)
    
    def decorator(func: Callable) -> Callable:
        return manager(func)
    
    return decorator


class AsyncRetryManager:
    """Async version of retry manager."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(f"async_retry_manager.{config.name or 'default'}")
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_attempts': 0,
            'max_attempts_used': 0
        }
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for async functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        return wrapper
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        import asyncio
        
        self.stats['total_calls'] += 1
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            self.stats['total_attempts'] += 1
            
            try:
                self.logger.debug(f"Async attempt {attempt}/{self.config.max_attempts}")
                result = await func(*args, **kwargs)
                
                self.stats['successful_calls'] += 1
                self.stats['max_attempts_used'] = max(
                    self.stats['max_attempts_used'], 
                    attempt
                )
                
                if attempt > 1:
                    self.logger.info(f"Async function succeeded on attempt {attempt}")
                
                return result
                
            except self.config.exceptions as e:
                last_exception = e
                self.logger.warning(f"Async attempt {attempt} failed: {e}")
                
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    self.logger.debug(f"Async retrying in {delay:.2f} seconds")
                    await asyncio.sleep(delay)
            
            except Exception as e:
                self.stats['failed_calls'] += 1
                self.logger.error(f"Non-retryable async exception: {e}")
                raise
        
        self.stats['failed_calls'] += 1
        self.stats['max_attempts_used'] = max(
            self.stats['max_attempts_used'], 
            self.config.max_attempts
        )
        
        raise RetryExhaustedError(self.config.max_attempts, last_exception)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay (same as sync version)."""
        # Reuse logic from RetryManager
        sync_manager = RetryManager(self.config)
        return sync_manager._calculate_delay(attempt)


def async_retry(max_attempts: int = 3,
                base_delay: float = 1.0,
                max_delay: float = 60.0,
                strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
                jitter: bool = True,
                exceptions: Union[Type[Exception], tuple] = Exception,
                backoff_multiplier: float = 2.0,
                name: Optional[str] = None) -> Callable:
    """Decorator for async retry logic."""
    if isinstance(exceptions, type):
        exceptions = (exceptions,)
    
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=strategy,
        jitter=jitter,
        exceptions=exceptions,
        backoff_multiplier=backoff_multiplier,
        name=name
    )
    
    manager = AsyncRetryManager(config)
    
    def decorator(func: Callable) -> Callable:
        return manager(func)
    
    return decorator


class RetryableOperation:
    """Context manager for retryable operations."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.manager = RetryManager(config)
        self.attempt = 0
        self.max_attempts = config.max_attempts
        
    def __enter__(self):
        self.attempt += 1
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success
            return True
        
        if not isinstance(exc_val, self.config.exceptions):
            # Non-retryable exception
            return False
        
        if self.attempt >= self.max_attempts:
            # Exhausted attempts, let the exception propagate
            return False
        
        # Calculate delay and sleep
        delay = self.manager._calculate_delay(self.attempt)
        time.sleep(delay)
        
        # Suppress the exception to continue with next attempt
        return True


# Specialized retry configurations

class DatabaseRetryConfig(RetryConfig):
    """Retry configuration for database operations."""
    
    def __init__(self, **kwargs):
        defaults = {
            'max_attempts': 3,
            'base_delay': 0.5,
            'max_delay': 30.0,
            'strategy': RetryStrategy.EXPONENTIAL_BACKOFF,
            'jitter': True,
            'exceptions': (Exception,),  # Would be database-specific exceptions
            'backoff_multiplier': 2.0,
            'name': 'database'
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class NetworkRetryConfig(RetryConfig):
    """Retry configuration for network operations."""
    
    def __init__(self, **kwargs):
        defaults = {
            'max_attempts': 5,
            'base_delay': 1.0,
            'max_delay': 60.0,
            'strategy': RetryStrategy.EXPONENTIAL_BACKOFF,
            'jitter': True,
            'exceptions': (Exception,),  # Would be network-specific exceptions
            'backoff_multiplier': 2.0,
            'name': 'network'
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class ComputeRetryConfig(RetryConfig):
    """Retry configuration for compute operations."""
    
    def __init__(self, **kwargs):
        defaults = {
            'max_attempts': 2,
            'base_delay': 5.0,
            'max_delay': 300.0,
            'strategy': RetryStrategy.FIXED_DELAY,
            'jitter': False,
            'exceptions': (Exception,),  # Would be compute-specific exceptions
            'backoff_multiplier': 1.0,
            'name': 'compute'
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


# Convenience decorators

database_retry = lambda func: retry(
    max_attempts=3,
    base_delay=0.5,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    name='database'
)(func)

network_retry = lambda func: retry(
    max_attempts=5,
    base_delay=1.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    name='network'
)(func)

compute_retry = lambda func: retry(
    max_attempts=2,
    base_delay=5.0,
    strategy=RetryStrategy.FIXED_DELAY,
    name='compute'
)(func)