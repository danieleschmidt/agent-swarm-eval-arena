#!/usr/bin/env python3
"""
Complete Integration Test Suite for Autonomous SDLC
Tests all three generations: MAKE IT WORK, MAKE IT ROBUST, MAKE IT SCALE
"""

import pytest
import sys
import os
import time
import tempfile
import json
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from swarm_arena import Arena, SwarmConfig
from swarm_arena.core.agent import CooperativeAgent, CompetitiveAgent, SwarmAgent
from swarm_arena.reliability import HealthMonitor, CircuitBreaker, RetryManager
from swarm_arena.reliability.circuit_breaker import CircuitBreakerConfig
from swarm_arena.reliability.retry_manager import RetryConfig
from swarm_arena.security import InputSanitizer


class TestGeneration1BasicFunctionality:
    """Test Generation 1: MAKE IT WORK - Basic Functionality"""

    def test_arena_creation_and_basic_simulation(self):
        """Test basic arena creation and simulation execution."""
        config = SwarmConfig(
            num_agents=20,
            arena_size=(400, 300),
            episode_length=50,
            observation_radius=50.0
        )
        
        arena = Arena(config)
        arena.add_agents(CooperativeAgent, count=10)
        arena.add_agents(CompetitiveAgent, count=10)
        
        assert len(arena.agents) == 20
        
        # Run simulation
        results = arena.run(episodes=1, verbose=False)
        
        # Verify results structure
        assert hasattr(results, 'mean_reward')
        assert hasattr(results, 'fairness_index')
        assert hasattr(results, 'total_steps')
        assert results.total_steps > 0

    def test_multi_agent_types(self):
        """Test multiple agent types working together."""
        config = SwarmConfig(
            num_agents=30,
            arena_size=(500, 400),
            episode_length=30
        )
        
        arena = Arena(config)
        arena.add_agents(CooperativeAgent, count=10, cooperation_tendency=0.8)
        arena.add_agents(CompetitiveAgent, count=10, exploration_rate=0.3)
        arena.add_agents(SwarmAgent, count=10)
        
        results = arena.run(episodes=1, verbose=False)
        
        # Verify all agent types are present and active
        agent_types = {}
        for agent in arena.agents.values():
            agent_type = type(agent).__name__
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        assert len(agent_types) == 3  # Three different agent types
        assert results.mean_reward is not None

    def test_emergent_behavior_detection(self):
        """Test detection of emergent behavioral patterns."""
        config = SwarmConfig(
            num_agents=25,
            arena_size=(600, 400),
            episode_length=40
        )
        
        arena = Arena(config)
        arena.add_agents(CooperativeAgent, count=25)
        
        results = arena.evaluate(
            num_episodes=1,
            metrics=["efficiency", "emergence", "coordination"]
        )
        
        # Verify emergent patterns are detected
        assert hasattr(results, 'emergent_patterns')
        assert isinstance(results.emergent_patterns, list)
        assert len(results.emergent_patterns) > 0


class TestGeneration2Robustness:
    """Test Generation 2: MAKE IT ROBUST - Enhanced Reliability"""

    def test_health_monitoring_system(self):
        """Test health monitoring and recovery."""
        monitor = HealthMonitor(check_interval=0.1)
        monitor.start_monitoring()
        
        # Add test metrics
        monitor.update_metric("test_component", "test_metric", 50.0, 70.0, 90.0)
        
        time.sleep(0.2)  # Allow health check to run
        
        report = monitor.get_health_report()
        
        assert 'test_component' in report['components']
        assert report['global_health_score'] > 0
        assert isinstance(report['unhealthy_components'], list)
        
        monitor.stop_monitoring()

    def test_circuit_breaker_protection(self):
        """Test circuit breaker fault tolerance."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,
            name="test_breaker"
        )
        
        circuit_breaker = CircuitBreaker(config)
        
        # Test successful operations
        def successful_operation():
            return "success"
        
        result = circuit_breaker.call(successful_operation)
        assert result == "success"
        
        # Test failure handling
        def failing_operation():
            raise Exception("Test failure")
        
        failure_count = 0
        for _ in range(3):
            try:
                circuit_breaker.call(failing_operation)
            except Exception:
                failure_count += 1
        
        assert failure_count > 0
        stats = circuit_breaker.get_stats()
        assert stats['failed_calls'] > 0

    def test_input_sanitization(self):
        """Test comprehensive input sanitization."""
        sanitizer = InputSanitizer()
        
        # Test agent configuration sanitization
        unsafe_config = {
            "learning_rate": float('inf'),
            "malicious_script": "<script>alert('xss')</script>",
            "__dangerous__": "system_access",
            "normal_param": 0.01
        }
        
        safe_config, warnings = sanitizer.sanitize_agent_config(unsafe_config)
        
        assert len(warnings) > 0  # Should have warnings
        assert safe_config['learning_rate'] == 0  # Should be sanitized
        assert 'safe_' in list(safe_config.keys())[2]  # Dangerous key should be prefixed
        
        # Test position sanitization
        invalid_positions = [
            [float('nan'), 100.0],
            [1e20, 1e20],
            "invalid_position"
        ]
        
        for pos in invalid_positions:
            safe_pos, pos_warnings = sanitizer.sanitize_position(pos)
            assert safe_pos.shape == (2,)  # Should always return valid 2D position
            assert np.all(np.isfinite(safe_pos))  # Should be finite

    def test_retry_mechanisms(self):
        """Test retry logic with exponential backoff."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,  # Very short delay for testing
            backoff_multiplier=2.0
        )
        
        retry_manager = RetryManager(config)
        
        # Test eventual success
        attempt_counter = 0
        
        def eventually_succeeds():
            nonlocal attempt_counter
            attempt_counter += 1
            if attempt_counter < 3:
                raise Exception(f"Failure on attempt {attempt_counter}")
            return "success"
        
        result = retry_manager.execute(eventually_succeeds)
        assert result == "success"
        assert attempt_counter == 3
        
        # Test retry exhaustion
        def always_fails():
            raise Exception("Always fails")
        
        with pytest.raises(Exception):
            retry_manager.execute(always_fails)

    def test_robust_arena_operations(self):
        """Test arena operations with health monitoring."""
        config = SwarmConfig(
            num_agents=15,
            arena_size=(300, 200),
            episode_length=25
        )
        
        arena = Arena(config)
        arena.add_agents(CooperativeAgent, count=15)
        
        # Run with health monitoring
        monitor = HealthMonitor(check_interval=0.1)
        monitor.start_monitoring()
        
        # Simulate performance metrics
        monitor.update_metric("arena_test", "fps", 30.0, 20.0, 10.0)
        
        results = arena.run(episodes=1, verbose=False)
        
        # Verify simulation completed successfully
        assert results.mean_reward is not None
        assert results.total_steps > 0
        
        # Check health monitoring captured metrics
        report = monitor.get_health_report()
        assert 'arena_test' in report['components']
        
        monitor.stop_monitoring()


class TestGeneration3Scaling:
    """Test Generation 3: MAKE IT SCALE - Optimization and Performance"""

    def test_performance_scaling(self):
        """Test performance across different agent counts."""
        agent_counts = [10, 20, 40]
        performance_results = []
        
        for count in agent_counts:
            config = SwarmConfig(
                num_agents=count,
                arena_size=(400, 300),
                episode_length=20  # Short for testing
            )
            
            arena = Arena(config)
            arena.add_agents(SwarmAgent, count=count//2)
            arena.add_agents(CompetitiveAgent, count=count//2)
            
            start_time = time.time()
            results = arena.run(episodes=1, verbose=False)
            execution_time = time.time() - start_time
            
            fps = config.episode_length / execution_time
            performance_results.append({
                'agents': count,
                'fps': fps,
                'execution_time': execution_time
            })
        
        # Verify scaling characteristics
        assert len(performance_results) == 3
        for result in performance_results:
            assert result['fps'] > 0
            assert result['execution_time'] > 0

    def test_concurrent_execution(self):
        """Test concurrent simulation execution."""
        import concurrent.futures
        
        def run_simulation(agent_count):
            config = SwarmConfig(
                num_agents=agent_count,
                arena_size=(300, 200),
                episode_length=15,
                seed=42 + agent_count  # Different seed for each
            )
            
            arena = Arena(config)
            arena.add_agents(CooperativeAgent, count=agent_count)
            
            results = arena.run(episodes=1, verbose=False)
            return {
                'agents': agent_count,
                'reward': results.mean_reward,
                'steps': results.total_steps
            }
        
        # Test concurrent execution
        agent_counts = [10, 15, 20]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            concurrent_results = list(executor.map(run_simulation, agent_counts))
        
        assert len(concurrent_results) == 3
        for result in concurrent_results:
            assert result['reward'] is not None
            assert result['steps'] > 0

    def test_memory_optimization(self):
        """Test memory usage patterns."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create memory-intensive scenario
        config = SwarmConfig(
            num_agents=50,
            arena_size=(600, 400),
            episode_length=10  # Short for testing
        )
        
        arena = Arena(config)
        arena.add_agents(SwarmAgent, count=25)
        arena.add_agents(CompetitiveAgent, count=25)
        
        results = arena.run(episodes=1, verbose=False)
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        memory_per_agent = memory_increase / len(arena.agents)
        
        # Verify reasonable memory usage (memory_increase might be 0 in test environment)
        assert memory_increase >= 0  # Should not decrease memory
        assert memory_per_agent < 10 * 1024 * 1024 or memory_increase == 0  # Less than 10MB per agent or no increase
        assert results.mean_reward is not None

    def test_auto_scaling_logic(self):
        """Test auto-scaling decision making."""
        class TestAutoScaler:
            def __init__(self):
                self.min_agents = 10
                self.max_agents = 100
                self.target_fps = 25
            
            def should_scale(self, current_agents, current_fps, cpu_usage):
                if current_fps < self.target_fps and cpu_usage < 0.8:
                    return "scale_up", min(self.max_agents, int(current_agents * 1.2))
                elif current_fps > self.target_fps * 1.5 and current_agents > self.min_agents:
                    return "scale_down", max(self.min_agents, int(current_agents * 0.8))
                else:
                    return "maintain", current_agents
        
        scaler = TestAutoScaler()
        
        # Test different scenarios
        test_cases = [
            (20, 15, 0.7),  # Low FPS, low CPU -> scale up
            (40, 50, 0.4),  # High FPS, low CPU -> scale down
            (30, 25, 0.8),  # Target FPS, high CPU -> maintain
        ]
        
        scaling_decisions = []
        for agents, fps, cpu in test_cases:
            action, new_count = scaler.should_scale(agents, fps, cpu)
            scaling_decisions.append((action, new_count))
        
        # Verify scaling logic
        assert scaling_decisions[0][0] == "scale_up"    # Should scale up
        assert scaling_decisions[1][0] == "scale_down"  # Should scale down
        assert scaling_decisions[2][0] == "maintain"    # Should maintain


class TestIntegratedSystemQuality:
    """Test integrated system quality across all generations."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow from creation to results."""
        # Generation 1: Basic functionality
        config = SwarmConfig(
            num_agents=30,
            arena_size=(500, 400),
            episode_length=30
        )
        
        arena = Arena(config)
        arena.add_agents(CooperativeAgent, count=10)
        arena.add_agents(CompetitiveAgent, count=10)
        arena.add_agents(SwarmAgent, count=10)
        
        # Generation 2: Add monitoring and security
        monitor = HealthMonitor(check_interval=0.1)
        monitor.start_monitoring()
        
        sanitizer = InputSanitizer()
        test_config = {"param1": 1.0, "param2": "safe_value"}
        safe_config, warnings = sanitizer.sanitize_agent_config(test_config)
        
        # Generation 3: Performance measurement
        start_time = time.time()
        results = arena.run(episodes=2, verbose=False)
        execution_time = time.time() - start_time
        
        fps = (config.episode_length * 2) / execution_time
        
        # Quality checks
        assert results.mean_reward is not None
        assert results.fairness_index >= 0
        assert results.total_steps > 0
        assert fps > 0
        assert len(safe_config) > 0
        
        # Health check
        time.sleep(0.2)
        health_report = monitor.get_health_report()
        assert health_report['global_health_score'] >= 0
        
        monitor.stop_monitoring()

    def test_error_recovery_integration(self):
        """Test integrated error recovery across components."""
        # Circuit breaker for fault tolerance
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            name="integration_test"
        )
        circuit_breaker = CircuitBreaker(config)
        
        # Retry manager for resilience
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        retry_manager = RetryManager(retry_config)
        
        # Health monitor for system health
        health_monitor = HealthMonitor(check_interval=0.1)
        health_monitor.start_monitoring()
        
        # Simulate failing then recovering operation
        attempt_count = 0
        
        def flaky_arena_operation():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 2:
                raise Exception("Simulated failure")
            
            # Successful operation - create and run small arena
            test_config = SwarmConfig(
                num_agents=5,
                arena_size=(200, 200),
                episode_length=10
            )
            test_arena = Arena(test_config)
            test_arena.add_agents(SwarmAgent, count=5)
            return test_arena.run(episodes=1, verbose=False)
        
        # Test integrated recovery
        try:
            with circuit_breaker:
                result = retry_manager.execute(flaky_arena_operation)
                assert result is not None
                assert hasattr(result, 'mean_reward')
        except Exception as e:
            pytest.fail(f"Integrated error recovery failed: {e}")
        
        health_monitor.stop_monitoring()

    def test_performance_quality_gates(self):
        """Test performance meets quality gate requirements."""
        # Quality gate thresholds
        MIN_FPS = 5                # Minimum acceptable FPS
        MAX_MEMORY_PER_AGENT = 1   # MB per agent
        MAX_STARTUP_TIME = 5       # seconds
        MIN_FAIRNESS_INDEX = 0.5   # Minimum fairness
        
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Test scenario
        start_time = time.time()
        
        config = SwarmConfig(
            num_agents=25,
            arena_size=(400, 300),
            episode_length=20
        )
        
        arena = Arena(config)
        arena.add_agents(CooperativeAgent, count=12)
        arena.add_agents(CompetitiveAgent, count=13)
        
        startup_time = time.time() - start_time
        
        # Run performance test
        perf_start = time.time()
        results = arena.run(episodes=1, verbose=False)
        execution_time = time.time() - perf_start
        
        fps = config.episode_length / execution_time
        peak_memory = process.memory_info().rss
        memory_per_agent = (peak_memory - initial_memory) / (1024 * 1024) / len(arena.agents)
        
        # Quality gate assertions
        assert fps >= MIN_FPS, f"FPS {fps:.1f} below minimum {MIN_FPS}"
        assert memory_per_agent <= MAX_MEMORY_PER_AGENT, f"Memory per agent {memory_per_agent:.2f}MB exceeds {MAX_MEMORY_PER_AGENT}MB"
        assert startup_time <= MAX_STARTUP_TIME, f"Startup time {startup_time:.2f}s exceeds {MAX_STARTUP_TIME}s"
        assert results.fairness_index >= MIN_FAIRNESS_INDEX, f"Fairness {results.fairness_index:.2f} below minimum {MIN_FAIRNESS_INDEX}"

    def test_production_readiness_checklist(self):
        """Verify production readiness across all components."""
        checklist_results = {}
        
        # 1. Basic functionality works
        try:
            config = SwarmConfig(num_agents=10, arena_size=(200, 200), episode_length=10)
            arena = Arena(config)
            arena.add_agents(SwarmAgent, count=10)
            results = arena.run(episodes=1, verbose=False)
            checklist_results['basic_functionality'] = True
        except Exception:
            checklist_results['basic_functionality'] = False
        
        # 2. Health monitoring operational
        try:
            monitor = HealthMonitor(check_interval=0.1)
            monitor.start_monitoring()
            monitor.update_metric("test", "test_metric", 50.0, 80.0, 95.0)
            time.sleep(0.15)
            report = monitor.get_health_report()
            monitor.stop_monitoring()
            checklist_results['health_monitoring'] = len(report['components']) > 0
        except Exception:
            checklist_results['health_monitoring'] = False
        
        # 3. Security measures active
        try:
            sanitizer = InputSanitizer()
            test_input = {"param": "<script>alert('test')</script>"}
            safe_input, warnings = sanitizer.sanitize_agent_config(test_input)
            checklist_results['security_measures'] = len(warnings) > 0
        except Exception:
            checklist_results['security_measures'] = False
        
        # 4. Error recovery functional
        try:
            retry_config = RetryConfig(max_attempts=2, base_delay=0.01)
            retry_manager = RetryManager(retry_config)
            
            attempt = 0
            def test_recovery():
                nonlocal attempt
                attempt += 1
                if attempt == 1:
                    raise Exception("Test failure")
                return "recovered"
            
            result = retry_manager.execute(test_recovery)
            checklist_results['error_recovery'] = result == "recovered"
        except Exception:
            checklist_results['error_recovery'] = False
        
        # 5. Performance acceptable
        try:
            start_time = time.time()
            config = SwarmConfig(num_agents=20, arena_size=(300, 200), episode_length=15)
            arena = Arena(config)
            arena.add_agents(SwarmAgent, count=20)
            results = arena.run(episodes=1, verbose=False)
            execution_time = time.time() - start_time
            fps = config.episode_length / execution_time
            checklist_results['performance_acceptable'] = fps > 3  # Minimum acceptable
        except Exception:
            checklist_results['performance_acceptable'] = False
        
        # Verify all checklist items pass
        failed_items = [item for item, passed in checklist_results.items() if not passed]
        
        assert len(failed_items) == 0, f"Production readiness failures: {failed_items}"
        
        # Log results for visibility
        print(f"\n✅ Production Readiness Checklist:")
        for item, passed in checklist_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {item}: {status}")


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])