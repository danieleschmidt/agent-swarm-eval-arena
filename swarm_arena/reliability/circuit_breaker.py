"""Circuit breaker pattern for system reliability."""

import time
import threading
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import functools
import logging


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception
    name: Optional[str] = None


class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger(f"circuit_breaker.{config.name or 'default'}")
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'rejected_calls': 0,
            'state_changes': 0
        }
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenException: When circuit is open
        """
        with self.lock:
            self.stats['total_calls'] += 1
            
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._change_state(CircuitState.HALF_OPEN)
                else:
                    self.stats['rejected_calls'] += 1
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker is open for {self.config.name or 'function'}"
                    )
            
            # Try to execute the function
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.config.expected_exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful function execution."""
        self.stats['successful_calls'] += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self._change_state(CircuitState.CLOSED)
        
        # Reset failure count on success
        self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed function execution."""
        self.stats['failed_calls'] += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if (self.state == CircuitState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            self._change_state(CircuitState.OPEN)
        elif self.state == CircuitState.HALF_OPEN:
            self._change_state(CircuitState.OPEN)
    
    def _change_state(self, new_state: CircuitState) -> None:
        """Change circuit breaker state."""
        old_state = self.state
        self.state = new_state
        self.stats['state_changes'] += 1
        
        self.logger.info(f"Circuit breaker state changed: {old_state.value} -> {new_state.value}")
        
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self.lock:
            return {
                **self.stats,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'last_failure_time': self.last_failure_time,
                'success_rate': (
                    self.stats['successful_calls'] / self.stats['total_calls']
                    if self.stats['total_calls'] > 0 else 0.0
                )
            }
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self.lock:
            self._change_state(CircuitState.CLOSED)
            self.failure_count = 0
            self.last_failure_time = None
    
    def force_open(self) -> None:
        """Manually force circuit breaker to open state."""
        with self.lock:
            self._change_state(CircuitState.OPEN)


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.Lock()
    
    def get_or_create(self, 
                     name: str, 
                     config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one.
        
        Args:
            name: Circuit breaker name
            config: Configuration (uses default if not provided)
            
        Returns:
            Circuit breaker instance
        """
        with self.lock:
            if name not in self.breakers:
                if config is None:
                    config = CircuitBreakerConfig(name=name)
                self.breakers[name] = CircuitBreaker(config)
            
            return self.breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers.
        
        Returns:
            Dictionary mapping names to statistics
        """
        with self.lock:
            return {name: breaker.get_stats() for name, breaker in self.breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self.lock:
            for breaker in self.breakers.values():
                breaker.reset()


# Global registry instance
circuit_registry = CircuitBreakerRegistry()


def circuit_breaker(name: str, 
                   failure_threshold: int = 5,
                   recovery_timeout: float = 60.0,
                   expected_exception: type = Exception) -> Callable:
    """Decorator to add circuit breaker protection to a function.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type that triggers circuit breaker
        
    Returns:
        Decorated function
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
        name=name
    )
    
    breaker = circuit_registry.get_or_create(name, config)
    
    def decorator(func: Callable) -> Callable:
        return breaker(func)
    
    return decorator


# Specialized circuit breakers for common use cases

class DatabaseCircuitBreaker(CircuitBreaker):
    """Circuit breaker specialized for database operations."""
    
    def __init__(self, name: str = "database"):
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=Exception,  # Would be database-specific exceptions
            name=name
        )
        super().__init__(config)


class APICircuitBreaker(CircuitBreaker):
    """Circuit breaker specialized for API calls."""
    
    def __init__(self, name: str = "api"):
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=Exception,  # Would be HTTP exceptions
            name=name
        )
        super().__init__(config)


class ComputeCircuitBreaker(CircuitBreaker):
    """Circuit breaker specialized for compute operations."""
    
    def __init__(self, name: str = "compute"):
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=120.0,
            expected_exception=Exception,  # Memory, timeout exceptions
            name=name
        )
        super().__init__(config)


class HealthCheckManager:
    """Manages health checks for various system components."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def register_health_check(self, 
                            name: str, 
                            check_func: Callable[[], bool],
                            timeout: float = 5.0) -> None:
        """Register a health check function.
        
        Args:
            name: Health check name
            check_func: Function that returns True if healthy
            timeout: Timeout for health check in seconds
        """
        with self.lock:
            self.health_checks[name] = {
                'func': check_func,
                'timeout': timeout
            }
            
            # Initialize status
            self.health_status[name] = {
                'healthy': True,
                'last_check': None,
                'last_error': None,
                'consecutive_failures': 0
            }
    
    def check_health(self, name: str) -> Dict[str, Any]:
        """Perform health check for a specific component.
        
        Args:
            name: Component name
            
        Returns:
            Health status dictionary
        """
        if name not in self.health_checks:
            return {'healthy': False, 'error': 'Health check not registered'}
        
        check_info = self.health_checks[name]
        
        try:
            # Execute health check with timeout
            result = self._execute_with_timeout(
                check_info['func'], 
                check_info['timeout']
            )
            
            with self.lock:
                self.health_status[name].update({
                    'healthy': result,
                    'last_check': time.time(),
                    'last_error': None,
                    'consecutive_failures': 0 if result else self.health_status[name]['consecutive_failures'] + 1
                })
                
                return self.health_status[name].copy()
        
        except Exception as e:
            with self.lock:
                self.health_status[name].update({
                    'healthy': False,
                    'last_check': time.time(),
                    'last_error': str(e),
                    'consecutive_failures': self.health_status[name]['consecutive_failures'] + 1
                })
                
                return self.health_status[name].copy()
    
    def check_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all registered components.
        
        Returns:
            Dictionary mapping component names to health status
        """
        results = {}
        
        for name in self.health_checks:
            results[name] = self.check_health(name)
        
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status.
        
        Returns:
            Overall health summary
        """
        all_health = self.check_all_health()
        
        healthy_count = sum(1 for status in all_health.values() if status['healthy'])
        total_count = len(all_health)
        
        overall_healthy = healthy_count == total_count
        
        return {
            'healthy': overall_healthy,
            'healthy_components': healthy_count,
            'total_components': total_count,
            'health_score': healthy_count / total_count if total_count > 0 else 1.0,
            'components': all_health,
            'timestamp': time.time()
        }
    
    def _execute_with_timeout(self, func: Callable, timeout: float) -> bool:
        """Execute function with timeout."""
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def worker():
            try:
                result = func()
                result_queue.put(result)
            except Exception as e:
                result_queue.put(e)
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError(f"Health check timed out after {timeout} seconds")
        
        try:
            result = result_queue.get_nowait()
            if isinstance(result, Exception):
                raise result
            return result
        except queue.Empty:
            raise TimeoutError("Health check did not complete")


# Default health check functions for common components

def memory_health_check() -> bool:
    """Check if memory usage is within acceptable limits."""
    import psutil
    
    memory = psutil.virtual_memory()
    return memory.percent < 90.0  # Less than 90% memory usage


def disk_health_check() -> bool:
    """Check if disk space is available."""
    import psutil
    
    disk = psutil.disk_usage('/')
    return disk.percent < 95.0  # Less than 95% disk usage


def cpu_health_check() -> bool:
    """Check if CPU usage is within limits."""
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=1)
    return cpu_percent < 95.0  # Less than 95% CPU usage


# Global health check manager
health_manager = HealthCheckManager()

# Register default health checks
health_manager.register_health_check("memory", memory_health_check)
health_manager.register_health_check("disk", disk_health_check)
health_manager.register_health_check("cpu", cpu_health_check)