"""Resilience and fault tolerance tests for Swarm Arena."""

import pytest
import time
import threading
from unittest.mock import Mock, patch

from swarm_arena.resilience.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitBreakerManager,
    circuit_breaker, circuit_manager, CONFIGS
)
from swarm_arena.exceptions import CircuitBreakerError


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and initial state."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=10.0
        )
        
        cb = CircuitBreaker("test", config)
        
        assert cb.name == "test"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
    
    def test_successful_calls(self):
        """Test circuit breaker with successful calls."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test_success", config)
        
        def successful_function():
            return "success"
        
        # Multiple successful calls should work fine
        for _ in range(5):
            result = cb.call(successful_function)
            assert result == "success"
            assert cb.state == CircuitState.CLOSED
            assert cb.failure_count == 0
    
    def test_circuit_opens_on_failures(self):
        """Test that circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3, timeout=1.0)
        cb = CircuitBreaker("test_failures", config)
        
        def failing_function():
            raise RuntimeError("Test failure")
        
        # First two failures should not open circuit
        for i in range(2):
            with pytest.raises(RuntimeError):
                cb.call(failing_function)
            assert cb.state == CircuitState.CLOSED
            assert cb.failure_count == i + 1
        
        # Third failure should open circuit
        with pytest.raises(RuntimeError):
            cb.call(failing_function)
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3
    
    def test_circuit_fails_fast_when_open(self):
        """Test that circuit fails fast when open."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=10.0)
        cb = CircuitBreaker("test_open", config)
        
        def failing_function():
            raise RuntimeError("Test failure")
        
        # Trigger circuit opening
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(failing_function)
        
        assert cb.state == CircuitState.OPEN
        
        # Further calls should fail fast with CircuitBreakerError
        with pytest.raises(CircuitBreakerError, match="is OPEN"):
            cb.call(lambda: "should not execute")
    
    def test_circuit_half_open_transition(self):
        """Test transition to half-open state after timeout."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)  # Very short timeout
        cb = CircuitBreaker("test_half_open", config)
        
        def failing_function():
            raise RuntimeError("Test failure")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(failing_function)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.2)
        
        def successful_function():
            return "recovery"
        
        # Next call should transition to half-open and succeed
        result = cb.call(successful_function)
        assert result == "recovery"
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_circuit_closes_after_successes(self):
        """Test that circuit closes after enough successes in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=2, 
            success_threshold=2, 
            timeout=0.1
        )
        cb = CircuitBreaker("test_close", config)
        
        def failing_function():
            raise RuntimeError("Test failure")
        
        def successful_function():
            return "success"
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(failing_function)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for timeout and transition to half-open
        time.sleep(0.2)
        cb.call(successful_function)
        assert cb.state == CircuitState.HALF_OPEN
        
        # Another success should close the circuit
        cb.call(successful_function)
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_circuit_reopens_on_failure_in_half_open(self):
        """Test that circuit reopens on failure in half-open state."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        cb = CircuitBreaker("test_reopen", config)
        
        def failing_function():
            raise RuntimeError("Test failure")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(failing_function)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Failure in half-open should reopen circuit
        with pytest.raises(RuntimeError):
            cb.call(failing_function)
        
        assert cb.state == CircuitState.OPEN
    
    def test_slow_call_detection(self):
        """Test detection of slow calls."""
        config = CircuitBreakerConfig(
            slow_call_threshold=0.1,  # 100ms
            slow_call_rate_threshold=0.5  # 50%
        )
        cb = CircuitBreaker("test_slow", config)
        
        def slow_function():
            time.sleep(0.15)  # Slower than threshold
            return "slow"
        
        def fast_function():
            return "fast"
        
        # Mix of fast and slow calls
        for _ in range(10):
            cb.call(fast_function)
        
        for _ in range(15):  # Enough slow calls to trigger threshold
            cb.call(slow_function)
        
        # Circuit might open due to slow calls (depends on exact timing)
        stats = cb.get_stats()
        assert stats["slow_calls"] > 0
    
    def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics collection."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test_stats", config)
        
        def test_function():
            return "test"
        
        # Make some calls
        for _ in range(5):
            cb.call(test_function)
        
        stats = cb.get_stats()
        
        assert stats["name"] == "test_stats"
        assert stats["state"] == CircuitState.CLOSED.value
        assert stats["total_calls"] == 5
        assert stats["successful_calls"] == 5
        assert stats["failed_calls"] == 0
        assert stats["success_rate"] == 1.0
    
    def test_circuit_breaker_reset(self):
        """Test manual circuit breaker reset."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test_reset", config)
        
        def failing_function():
            raise RuntimeError("Test failure")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(failing_function)
        
        assert cb.state == CircuitState.OPEN
        
        # Reset should close circuit
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        
        # Should work normally after reset
        def successful_function():
            return "success"
        
        result = cb.call(successful_function)
        assert result == "success"


class TestCircuitBreakerManager:
    """Test circuit breaker manager functionality."""
    
    def test_manager_create_and_get(self):
        """Test creating and retrieving circuit breakers."""
        manager = CircuitBreakerManager()
        
        config = CircuitBreakerConfig(failure_threshold=3)
        cb1 = manager.get_or_create("test1", config)
        
        # Getting same name should return same instance
        cb2 = manager.get_or_create("test1")
        assert cb1 is cb2
        
        # Different name should create new instance
        cb3 = manager.get_or_create("test2")
        assert cb3 is not cb1
    
    def test_manager_get_all_stats(self):
        """Test getting statistics for all circuit breakers."""
        manager = CircuitBreakerManager()
        
        cb1 = manager.get_or_create("cb1")
        cb2 = manager.get_or_create("cb2")
        
        def test_func():
            return "test"
        
        # Make some calls
        cb1.call(test_func)
        cb2.call(test_func)
        
        all_stats = manager.get_all_stats()
        assert "cb1" in all_stats
        assert "cb2" in all_stats
        assert all_stats["cb1"]["total_calls"] == 1
        assert all_stats["cb2"]["total_calls"] == 1
    
    def test_manager_reset_all(self):
        """Test resetting all circuit breakers."""
        manager = CircuitBreakerManager()
        
        # Create some circuit breakers and trigger failures
        for i in range(3):
            cb = manager.get_or_create(f"cb{i}", CircuitBreakerConfig(failure_threshold=1))
            
            def failing_func():
                raise RuntimeError("fail")
            
            with pytest.raises(RuntimeError):
                cb.call(failing_func)
        
        # All should be open
        unhealthy = manager.get_unhealthy_circuits()
        assert len(unhealthy) == 3
        
        # Reset all
        manager.reset_all()
        
        # All should be closed
        unhealthy = manager.get_unhealthy_circuits()
        assert len(unhealthy) == 0
    
    def test_manager_remove_circuit(self):
        """Test removing circuit breakers."""
        manager = CircuitBreakerManager()
        
        cb = manager.get_or_create("to_remove")
        assert "to_remove" in manager.circuit_breakers
        
        # Remove circuit breaker
        removed = manager.remove("to_remove")
        assert removed is True
        assert "to_remove" not in manager.circuit_breakers
        
        # Removing non-existent should return False
        removed = manager.remove("nonexistent")
        assert removed is False


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator functionality."""
    
    def test_decorator_basic_usage(self):
        """Test basic decorator usage."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        
        @circuit_breaker("decorator_test", config)
        def test_function(x):
            if x < 0:
                raise ValueError("Negative value")
            return x * 2
        
        # Successful calls should work
        assert test_function(5) == 10
        assert test_function(0) == 0
        
        # Check that circuit breaker was created
        cb = circuit_manager.circuit_breakers.get("decorator_test")
        assert cb is not None
        assert cb.config.failure_threshold == 2
    
    def test_decorator_with_failures(self):
        """Test decorator behavior with failures."""
        @circuit_breaker("decorator_fail", CircuitBreakerConfig(failure_threshold=2))
        def failing_function():
            raise RuntimeError("Always fails")
        
        # First two failures
        for _ in range(2):
            with pytest.raises(RuntimeError):
                failing_function()
        
        # Third call should trigger circuit breaker
        with pytest.raises(CircuitBreakerError):
            failing_function()
    
    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        @circuit_breaker("metadata_test")
        def documented_function(x, y=None):
            """This function has documentation."""
            return x
        
        assert documented_function.__name__ == "documented_function"
        assert "documentation" in documented_function.__doc__
        
        # Function should still work
        assert documented_function(42) == 42


class TestCircuitBreakerConcurrency:
    """Test circuit breaker behavior under concurrent access."""
    
    def test_concurrent_calls(self):
        """Test circuit breaker with concurrent calls."""
        config = CircuitBreakerConfig(failure_threshold=5, timeout=0.1)
        cb = CircuitBreaker("concurrent_test", config)
        
        results = []
        errors = []
        
        def worker_function(worker_id):
            try:
                def test_func():
                    time.sleep(0.01)  # Small delay
                    return f"worker_{worker_id}"
                
                result = cb.call(test_func)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple workers concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All calls should succeed
        assert len(results) == 10
        assert len(errors) == 0
        
        # Check statistics
        stats = cb.get_stats()
        assert stats["total_calls"] == 10
        assert stats["successful_calls"] == 10
    
    def test_concurrent_failures_and_successes(self):
        """Test circuit breaker with mixed concurrent calls."""
        config = CircuitBreakerConfig(failure_threshold=3, timeout=0.1)
        cb = CircuitBreaker("mixed_concurrent", config)
        
        results = []
        errors = []
        
        def worker_function(worker_id, should_fail):
            try:
                def test_func():
                    if should_fail:
                        raise RuntimeError(f"Worker {worker_id} failed")
                    return f"worker_{worker_id}_success"
                
                result = cb.call(test_func)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Mix of successful and failing workers
        threads = []
        for i in range(8):
            should_fail = i < 2  # First 2 workers fail
            thread = threading.Thread(target=worker_function, args=(i, should_fail))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have some successes and some failures
        assert len(results) > 0  # Some successful calls
        assert len(errors) > 0   # Some failed calls
        
        # Circuit should still be closed (not enough failures)
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with other components."""
    
    def test_circuit_breaker_with_simulation(self):
        """Test circuit breaker protecting simulation functions."""
        from swarm_arena import Arena, SwarmConfig, Agent
        
        @circuit_breaker("simulation_protection", CONFIGS["simulation"])
        def protected_simulation_step(arena):
            # Simulate occasional failures
            import random
            if random.random() < 0.1:  # 10% failure rate
                raise RuntimeError("Simulation step failed")
            
            return arena.step()
        
        config = SwarmConfig(num_agents=10, episode_length=50, seed=42)
        arena = Arena(config)
        arena.add_agents(Agent, count=10)
        arena.reset()
        
        successful_steps = 0
        circuit_breaker_errors = 0
        other_errors = 0
        
        # Run many steps
        for _ in range(100):
            try:
                protected_simulation_step(arena)
                successful_steps += 1
            except CircuitBreakerError:
                circuit_breaker_errors += 1
                break  # Stop when circuit opens
            except Exception:
                other_errors += 1
        
        # Should have some successful steps
        assert successful_steps > 0
        
        # Circuit breaker should eventually activate if enough failures occur
        # (This is probabilistic, so we don't assert it always happens)
    
    def test_multiple_circuit_breakers_isolation(self):
        """Test that multiple circuit breakers work independently."""
        config1 = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        config2 = CircuitBreakerConfig(failure_threshold=3, timeout=0.1)
        
        cb1 = circuit_manager.get_or_create("isolated_1", config1)
        cb2 = circuit_manager.get_or_create("isolated_2", config2)
        
        def failing_func():
            raise RuntimeError("Failure")
        
        def success_func():
            return "Success"
        
        # Trigger failures in cb1 only
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb1.call(failing_func)
        
        # cb1 should be open, cb2 should be closed
        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.CLOSED
        
        # cb2 should still work
        result = cb2.call(success_func)
        assert result == "Success"
        
        # cb1 should fail fast
        with pytest.raises(CircuitBreakerError):
            cb1.call(success_func)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])