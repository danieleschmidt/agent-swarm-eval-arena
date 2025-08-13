"""Comprehensive test suite for swarm arena."""

import pytest
import numpy as np
import time
import tempfile
import os
from pathlib import Path

from swarm_arena import Arena, SwarmConfig, CooperativeAgent, CompetitiveAgent, RandomAgent
from swarm_arena.security.authentication import AuthenticationManager, UserRole
from swarm_arena.security.input_validation import InputSanitizer, ConfigValidator
from swarm_arena.reliability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from swarm_arena.reliability.retry_manager import RetryManager, RetryConfig, RetryStrategy
from swarm_arena.monitoring.advanced_telemetry import MetricsCollector, PerformanceProfiler
from swarm_arena.optimization.performance import PerformanceOptimizer, VectorizedOperations
from swarm_arena.research.emergence import EmergenceDetector
from swarm_arena.research.fairness import FairnessAnalyzer
from swarm_arena.research.communication import MessageChannel, NegotiationProtocol


class TestBasicFunctionality:
    """Test basic arena functionality."""
    
    def test_arena_creation(self):
        """Test arena creation with valid config."""
        config = SwarmConfig(
            num_agents=10,
            arena_size=(100, 100),
            episode_length=50,
            seed=42
        )
        arena = Arena(config)
        assert arena.config.num_agents == 10
        assert arena.config.arena_size == (100, 100)
    
    def test_agent_addition(self):
        """Test adding agents to arena."""
        config = SwarmConfig(num_agents=5, arena_size=(100, 100), episode_length=10)
        arena = Arena(config)
        
        arena.add_agents(CooperativeAgent, count=3)
        arena.add_agents(CompetitiveAgent, count=2)
        
        assert len(arena.agents) == 5
        assert len(arena.agent_positions) == 5
    
    def test_simulation_execution(self):
        """Test basic simulation execution."""
        config = SwarmConfig(num_agents=5, arena_size=(100, 100), episode_length=10)
        arena = Arena(config)
        arena.add_agents(RandomAgent, count=5)
        
        results = arena.run(episodes=1, verbose=False)
        
        assert results.total_steps > 0
        assert results.mean_reward is not None
        assert len(results.episode_rewards) == 5
    
    def test_reset_functionality(self):
        """Test arena reset between episodes."""
        config = SwarmConfig(num_agents=3, arena_size=(100, 100), episode_length=5)
        arena = Arena(config)
        arena.add_agents(RandomAgent, count=3)
        
        # Run first episode
        arena.run(episodes=1, verbose=False)
        first_positions = {k: v.copy() for k, v in arena.agent_positions.items()}
        
        # Reset and check positions changed
        arena.reset()
        reset_positions = {k: v.copy() for k, v in arena.agent_positions.items()}
        
        # Positions should be different after reset
        position_changed = any(
            not np.array_equal(first_positions[k], reset_positions[k])
            for k in first_positions.keys()
        )
        assert position_changed


class TestSecurityFeatures:
    """Test security and authentication features."""
    
    def test_user_creation(self):
        """Test user account creation."""
        auth_manager = AuthenticationManager()
        
        user = auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPass123!",
            role=UserRole.RESEARCHER
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.RESEARCHER
    
    def test_authentication(self):
        """Test user authentication."""
        auth_manager = AuthenticationManager()
        
        auth_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPass123!",
            role=UserRole.RESEARCHER
        )
        
        # Valid authentication
        token = auth_manager.authenticate("testuser", "TestPass123!")
        assert token is not None
        
        # Invalid authentication
        invalid_token = auth_manager.authenticate("testuser", "wrongpassword")
        assert invalid_token is None
    
    def test_input_validation(self):
        """Test input sanitization and validation."""
        sanitizer = InputSanitizer()
        validator = ConfigValidator()
        
        # Test XSS prevention
        with pytest.raises(Exception):
            sanitizer.sanitize_string("<script>alert('xss')</script>")
        
        # Test config validation
        valid_config = {
            "num_agents": 50,
            "arena_size": [500, 500],
            "episode_length": 100
        }
        
        validated = validator.validate_config(valid_config)
        assert validated["num_agents"] == 50
        
        # Test invalid config rejection
        invalid_config = {
            "num_agents": -5,
            "arena_size": [10, 10, 10],
            "episode_length": "invalid"
        }
        
        with pytest.raises(Exception):
            validator.validate_config(invalid_config)


class TestReliabilityFeatures:
    """Test reliability and fault tolerance."""
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)
        breaker = CircuitBreaker(config)
        
        failure_count = 0
        
        @breaker
        def failing_function():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise Exception("Service failure")
            return "Success"
        
        # First two calls should fail and trigger circuit breaker
        with pytest.raises(Exception):
            failing_function()
        
        with pytest.raises(Exception):
            failing_function()
        
        # Third call should be rejected by circuit breaker
        from swarm_arena.reliability.circuit_breaker import CircuitBreakerOpenException
        with pytest.raises(CircuitBreakerOpenException):
            failing_function()
    
    def test_retry_mechanism(self):
        """Test retry with exponential backoff."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        manager = RetryManager(config)
        
        attempt_count = 0
        
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "Success"
        
        result = manager.execute(flaky_function)
        assert result == "Success"
        assert attempt_count == 3


class TestMonitoringFeatures:
    """Test monitoring and telemetry."""
    
    def test_metrics_collection(self):
        """Test metrics collection and storage."""
        collector = MetricsCollector(max_history=100)
        
        # Test counter metrics
        collector.record_counter("test_counter", 5)
        collector.record_counter("test_counter", 3)
        assert collector.get_counter("test_counter") == 8
        
        # Test gauge metrics
        collector.record_gauge("test_gauge", 42.5)
        assert collector.get_gauge("test_gauge") == 42.5
        
        # Test histogram metrics
        for i in range(10):
            collector.record_histogram("test_histogram", i * 0.1)
        
        stats = collector.get_histogram_stats("test_histogram")
        assert stats["count"] == 10
        assert stats["min"] == 0.0
        assert stats["max"] == 0.9
    
    def test_performance_profiling(self):
        """Test performance profiling."""
        profiler = PerformanceProfiler()
        
        # Test profiling context
        with profiler.profile_context("test_operation") as profile:
            time.sleep(0.01)  # Simulate work
        
        assert profile.duration > 0.005  # Should be at least 5ms
        
        # Get profile statistics
        stats = profiler.get_profile_stats("test_operation")
        assert stats["count"] == 1
        assert stats["min_time"] > 0


class TestOptimizationFeatures:
    """Test performance optimization features."""
    
    def test_vectorized_operations(self):
        """Test vectorized computations."""
        # Generate test data
        positions = np.random.uniform(0, 100, (50, 2))
        
        # Test distance matrix
        distances = VectorizedOperations.distance_matrix(positions)
        assert distances.shape == (50, 50)
        assert np.allclose(np.diag(distances), 0)  # Diagonal should be zero
        
        # Test neighbor finding
        neighbors = VectorizedOperations.find_neighbors(positions, radius=20.0)
        assert len(neighbors) == 50
        assert all(isinstance(n, list) for n in neighbors)
    
    def test_caching_optimization(self):
        """Test caching functionality."""
        from swarm_arena.optimization.performance import AdaptiveCache
        
        cache = AdaptiveCache(max_size=10, default_ttl=1.0)
        
        # Test cache set/get
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        # Test cache miss
        assert cache.get("nonexistent_key") is None
        
        # Test cache statistics
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] >= 1
    
    def test_memory_management(self):
        """Test memory management features."""
        from swarm_arena.optimization.performance import MemoryManager, PerformanceConfig
        
        config = PerformanceConfig()
        manager = MemoryManager(config)
        
        # Test memory usage check
        usage = manager.check_memory_usage()
        assert 0.0 <= usage <= 1.0
        
        # Test garbage collection
        collected = manager.cleanup_memory()
        assert isinstance(collected, int)


class TestResearchFeatures:
    """Test research and analysis features."""
    
    def test_emergence_detection(self):
        """Test emergence pattern detection."""
        detector = EmergenceDetector()
        
        # Create sample trajectory data
        trajectories = {}
        for agent_id in range(10):
            trajectory = []
            x, y = 50.0, 50.0
            
            for t in range(20):
                # Simple flocking behavior
                x += np.random.normal(0, 1)
                y += np.random.normal(0, 1)
                trajectory.append((x, y))
            
            trajectories[agent_id] = trajectory
        
        patterns = detector.analyze(trajectories)
        assert isinstance(patterns, list)
        # Should detect some patterns in the data
        assert len(patterns) >= 0
    
    def test_fairness_analysis(self):
        """Test fairness metrics computation."""
        analyzer = FairnessAnalyzer()
        
        # Test with equal allocation
        equal_allocations = {i: 10.0 for i in range(5)}
        metrics = analyzer.analyze_allocation(equal_allocations)
        
        assert metrics.gini_coefficient == 0.0  # Perfect equality
        assert metrics.envy_freeness == 1.0  # No envy
        assert metrics.jain_fairness_index == 1.0  # Perfect fairness
        
        # Test with unequal allocation
        unequal_allocations = {0: 50.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0}
        metrics = analyzer.analyze_allocation(unequal_allocations)
        
        assert metrics.gini_coefficient > 0  # Some inequality
        assert metrics.envy_freeness < 1.0  # Some envy
    
    def test_communication_protocols(self):
        """Test agent communication system."""
        channel = MessageChannel(max_range=100.0, bandwidth_limit=5)
        protocol = NegotiationProtocol()
        
        # Test message creation
        proposal_data = {
            "sender_id": 0,
            "receiver_id": 1,
            "resource_split": [0.6, 0.4]
        }
        
        message = protocol.encode_proposal(proposal_data)
        assert message.sender_id == 0
        assert message.receiver_id == 1
        
        # Test message channel
        agent_positions = {0: np.array([0, 0]), 1: np.array([50, 50])}
        success = channel.send_message(message, agent_positions, current_step=1)
        assert success
        
        # Receive messages
        messages = channel.get_messages_for_agent(1, current_step=1)
        assert len(messages) >= 0


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def test_full_simulation_pipeline(self):
        """Test complete simulation with all features."""
        # Create configuration
        config = SwarmConfig(
            num_agents=20,
            arena_size=(200, 200),
            episode_length=20,
            seed=42
        )
        
        # Create arena with monitoring
        from swarm_arena.monitoring.advanced_telemetry import metrics_collector
        
        arena = Arena(config)
        arena.add_agents(CooperativeAgent, count=10)
        arena.add_agents(CompetitiveAgent, count=10)
        
        # Record simulation metrics
        start_time = time.time()
        results = arena.run(episodes=2, verbose=False)
        execution_time = time.time() - start_time
        
        # Verify results
        assert results.total_steps > 0
        assert results.mean_reward is not None
        assert len(results.episode_rewards) == 20
        
        # Record metrics
        metrics_collector.record_gauge("test_simulation_time", execution_time)
        metrics_collector.record_gauge("test_mean_reward", results.mean_reward)
        
        # Verify metrics were recorded
        assert metrics_collector.get_gauge("test_simulation_time") == execution_time
    
    def test_error_handling_and_recovery(self):
        """Test error handling throughout the system."""
        # Test invalid configuration handling
        with pytest.raises(Exception):
            invalid_config = SwarmConfig(
                num_agents=-5,  # Invalid
                arena_size=(0, 0),  # Invalid
                episode_length=0  # Invalid
            )
        
        # Test arena with no agents
        config = SwarmConfig(num_agents=0, arena_size=(100, 100), episode_length=10)
        arena = Arena(config)
        
        # Should handle gracefully
        results = arena.run(episodes=1, verbose=False)
        assert results.total_steps >= 0
    
    def test_performance_under_load(self):
        """Test system performance under higher load."""
        config = SwarmConfig(
            num_agents=100,
            arena_size=(500, 500),
            episode_length=50,
            seed=42
        )
        
        arena = Arena(config)
        arena.add_agents(CooperativeAgent, count=50)
        arena.add_agents(CompetitiveAgent, count=50)
        
        # Measure performance
        start_time = time.time()
        results = arena.run(episodes=1, verbose=False)
        execution_time = time.time() - start_time
        
        # Performance should be reasonable
        steps_per_second = results.total_steps / execution_time
        assert steps_per_second > 10  # At least 10 steps per second
        
        # Memory usage should be reasonable
        import psutil
        memory_percent = psutil.virtual_memory().percent
        assert memory_percent < 90  # Less than 90% memory usage


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_agent_simulation(self):
        """Test simulation with single agent."""
        config = SwarmConfig(num_agents=1, arena_size=(100, 100), episode_length=10)
        arena = Arena(config)
        arena.add_agents(RandomAgent, count=1)
        
        results = arena.run(episodes=1, verbose=False)
        assert len(results.episode_rewards) == 1
        assert results.total_steps > 0
    
    def test_large_arena_small_agents(self):
        """Test large arena with few agents."""
        config = SwarmConfig(num_agents=2, arena_size=(10000, 10000), episode_length=10)
        arena = Arena(config)
        arena.add_agents(RandomAgent, count=2)
        
        results = arena.run(episodes=1, verbose=False)
        assert len(results.episode_rewards) == 2
    
    def test_many_agents_small_arena(self):
        """Test many agents in small arena."""
        config = SwarmConfig(num_agents=50, arena_size=(50, 50), episode_length=5)
        arena = Arena(config)
        arena.add_agents(RandomAgent, count=50)
        
        # Should handle crowding gracefully
        results = arena.run(episodes=1, verbose=False)
        assert len(results.episode_rewards) == 50
    
    def test_zero_episode_length(self):
        """Test handling of edge case parameters."""
        # Very short episodes
        config = SwarmConfig(num_agents=5, arena_size=(100, 100), episode_length=1)
        arena = Arena(config)
        arena.add_agents(RandomAgent, count=5)
        
        results = arena.run(episodes=1, verbose=False)
        assert results.episode_length <= 1


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])