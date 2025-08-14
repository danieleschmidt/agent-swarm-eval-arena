"""Circuit breaker pattern implementation for resilient system operation."""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
from collections import deque
import statistics

from ..exceptions import CircuitBreakerError, SystemError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing fast
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes before closing
    timeout: float = 60.0              # Seconds before trying half-open
    window_size: int = 100             # Rolling window for statistics
    slow_call_threshold: float = 5.0   # Seconds to consider call slow
    slow_call_rate_threshold: float = 0.5  # Rate of slow calls to trip


@dataclass
class CallResult:
    """Result of a function call through circuit breaker."""
    success: bool
    duration: float
    timestamp: float
    error: Optional[Exception] = None


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.call_history: deque = deque(maxlen=config.window_size)
        self._lock = threading.RLock()
        
        logger.info(f"Created circuit breaker '{name}' with config: {config}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Last failure: {time.time() - self.last_failure_time:.1f}s ago"
                    )
            
            return self._execute_call(func, *args, **kwargs)
    
    def _execute_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the actual function call and handle results."""
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Record successful call
            call_result = CallResult(
                success=True,
                duration=duration,
                timestamp=start_time
            )
            self.call_history.append(call_result)
            
            # Handle success based on current state
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
            
            # Check for slow calls
            if duration > self.config.slow_call_threshold:
                logger.warning(f"Slow call detected in '{self.name}': {duration:.2f}s")
                self._check_slow_call_rate()
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record failed call
            call_result = CallResult(
                success=False,
                duration=duration,
                timestamp=start_time,
                error=e
            )
            self.call_history.append(call_result)
            
            self._handle_failure(e)
            raise
    
    def _handle_failure(self, error: Exception) -> None:
        """Handle call failure and update circuit state."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        logger.warning(f"Failure in circuit breaker '{self.name}': {error}")
        
        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state opens the circuit
            self._transition_to_open()
        elif self.state == CircuitState.CLOSED:
            # Check if we should open the circuit
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return time.time() - self.last_failure_time >= self.config.timeout
    
    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        previous_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}': {previous_state.value} -> CLOSED")
    
    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        previous_state = self.state
        self.state = CircuitState.OPEN
        self.success_count = 0
        logger.warning(f"Circuit breaker '{self.name}': {previous_state.value} -> OPEN")
    
    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        previous_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}': {previous_state.value} -> HALF_OPEN")
    
    def _check_slow_call_rate(self) -> None:
        """Check if slow call rate exceeds threshold."""
        if len(self.call_history) < 10:  # Need minimum calls for statistics
            return
        
        recent_calls = list(self.call_history)[-50:]  # Last 50 calls
        slow_calls = sum(1 for call in recent_calls 
                        if call.duration > self.config.slow_call_threshold)
        slow_call_rate = slow_calls / len(recent_calls)
        
        if slow_call_rate >= self.config.slow_call_rate_threshold:
            logger.warning(
                f"High slow call rate in '{self.name}': {slow_call_rate:.2%} "
                f"(threshold: {self.config.slow_call_rate_threshold:.2%})"
            )
            
            # Open circuit if too many slow calls
            if self.state == CircuitState.CLOSED:
                self._transition_to_open()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            recent_calls = list(self.call_history)
            
            if not recent_calls:
                return {
                    "name": self.name,
                    "state": self.state.value,
                    "failure_count": self.failure_count,
                    "success_count": self.success_count,
                    "total_calls": 0,
                    "success_rate": 0.0,
                    "average_duration": 0.0,
                    "last_failure_time": self.last_failure_time,
                }
            
            successful_calls = [call for call in recent_calls if call.success]
            failed_calls = [call for call in recent_calls if not call.success]
            
            durations = [call.duration for call in recent_calls]
            
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_calls": len(recent_calls),
                "successful_calls": len(successful_calls),
                "failed_calls": len(failed_calls),
                "success_rate": len(successful_calls) / len(recent_calls),
                "average_duration": statistics.mean(durations),
                "median_duration": statistics.median(durations),
                "p95_duration": self._percentile(durations, 95),
                "slow_calls": sum(1 for d in durations if d > self.config.slow_call_threshold),
                "last_failure_time": self.last_failure_time,
                "time_since_last_failure": time.time() - self.last_failure_time if self.last_failure_time > 0 else None,
            }
    
    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            previous_state = self.state
            self._transition_to_closed()
            self.call_history.clear()
            logger.info(f"Circuit breaker '{self.name}' manually reset from {previous_state.value}")
    
    def force_open(self) -> None:
        """Manually open circuit breaker."""
        with self._lock:
            previous_state = self.state
            self._transition_to_open()
            logger.info(f"Circuit breaker '{self.name}' manually opened from {previous_state.value}")
    
    @staticmethod
    def _percentile(data: list, percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self._lock:
            if name not in self.circuit_breakers:
                if config is None:
                    config = CircuitBreakerConfig()
                self.circuit_breakers[name] = CircuitBreaker(name, config)
            
            return self.circuit_breakers[name]
    
    def remove(self, name: str) -> bool:
        """Remove circuit breaker."""
        with self._lock:
            if name in self.circuit_breakers:
                del self.circuit_breakers[name]
                logger.info(f"Removed circuit breaker '{name}'")
                return True
            return False
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {name: cb.get_stats() 
                   for name, cb in self.circuit_breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for cb in self.circuit_breakers.values():
                cb.reset()
            logger.info("Reset all circuit breakers")
    
    def get_unhealthy_circuits(self) -> list:
        """Get list of circuit breakers that are not in CLOSED state."""
        with self._lock:
            return [name for name, cb in self.circuit_breakers.items() 
                   if cb.state != CircuitState.CLOSED]


# Global circuit breaker manager
circuit_manager = CircuitBreakerManager()


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to apply circuit breaker to a function."""
    def decorator(func: Callable) -> Callable:
        cb = circuit_manager.get_or_create(name, config)
        
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.circuit_breaker = cb
        
        return wrapper
    return decorator


# Predefined circuit breaker configurations
CONFIGS = {
    "simulation": CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=30.0,
        window_size=50,
        slow_call_threshold=10.0,
        slow_call_rate_threshold=0.3
    ),
    "distributed": CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=3,
        timeout=60.0,
        window_size=100,
        slow_call_threshold=30.0,
        slow_call_rate_threshold=0.4
    ),
    "database": CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=20.0,
        window_size=30,
        slow_call_threshold=5.0,
        slow_call_rate_threshold=0.2
    ),
}