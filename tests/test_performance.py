"""Performance tests for Swarm Arena."""

import pytest
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from swarm_arena import Arena, SwarmConfig, Agent, CooperativeAgent
from swarm_arena.optimization.performance_engine import (
    PerformanceOptimizer, OptimizationConfig, AdaptiveBatchProcessor,
    MemoryPool, VectorizedOperations, performance_optimizer
)
from swarm_arena.scaling.auto_scaler import (
    AutoScaler, ScalingEngine, ScalingPolicy, DEFAULT_POLICIES
)


class TestPerformanceOptimizer:
    """Test performance optimization engine."""
    
    def test_adaptive_batch_processor(self):
        """Test adaptive batching functionality."""
        config = OptimizationConfig(
            batch_size_min=10,
            batch_size_max=100,
            cpu_threshold=0.5,
            memory_threshold=0.5
        )
        
        processor = AdaptiveBatchProcessor(config)
        
        # Test batch processing
        items = list(range(50))
        
        def simple_processor(batch):
            return [x * 2 for x in batch]
        
        results = processor.process_batch(items, simple_processor)
        
        assert len(results) == 50
        assert results[0] == 0
        assert results[49] == 98
    
    def test_memory_pool(self):
        """Test memory pool for object reuse."""
        pool = MemoryPool(np.ndarray, initial_size=10, max_size=50)
        
        # Acquire objects
        objects = []
        for _ in range(20):
            obj = pool.acquire()
            assert isinstance(obj, np.ndarray)
            objects.append(obj)
        
        # Release objects
        for obj in objects[:10]:
            pool.release(obj)
        
        # Check stats
        stats = pool.get_stats()
        assert stats['available'] == 10
        assert stats['in_use'] == 10
    
    def test_vectorized_operations(self):
        """Test vectorized computation performance."""
        # Test distance computation
        positions = [(0, 0), (1, 1), (2, 2)]
        targets = [(0, 1), (1, 0)]
        
        distances = VectorizedOperations.compute_distances_vectorized(
            tuple(positions), tuple(targets)
        )
        
        assert distances.shape == (3, 2)
        assert np.isclose(distances[0, 0], 1.0)  # Distance from (0,0) to (0,1)
        assert np.isclose(distances[0, 1], 1.0)  # Distance from (0,0) to (1,0)
    
    def test_batch_agent_updates(self):
        """Test vectorized agent updates."""
        positions = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
        velocities = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.float32)
        actions = np.array([1, 2, 4])  # move_up, move_down, move_right
        
        new_positions, new_velocities = VectorizedOperations.batch_agent_updates(
            positions, velocities, actions, dt=1.0
        )
        
        assert new_positions.shape == (3, 2)
        assert new_velocities.shape == (3, 2)
        
        # Check specific movements
        assert np.array_equal(new_velocities[0], [0, 1])  # move_up
        assert np.array_equal(new_velocities[1], [0, -1])  # move_down
        assert np.array_equal(new_velocities[2], [1, 0])  # move_right
    
    def test_spatial_hash_vectorized(self):
        """Test vectorized spatial hashing."""
        positions = np.array([
            [0, 0], [50, 50], [100, 100], [55, 45], [105, 95]
        ], dtype=np.float32)
        
        spatial_hash = VectorizedOperations.spatial_hash_vectorized(positions, cell_size=50.0)
        
        # Agents in similar grid cells should be grouped
        assert isinstance(spatial_hash, dict)
        assert len(spatial_hash) <= 5  # At most 5 different cells
    
    def test_performance_optimizer_integration(self):
        """Test full performance optimizer integration."""
        config = OptimizationConfig(
            enable_adaptive_batching=True,
            enable_memory_pooling=True,
            enable_vectorization=True,
            max_worker_threads=4
        )
        
        optimizer = PerformanceOptimizer(config)
        optimizer.start_monitoring()
        
        try:
            # Test optimization of simulation step
            def dummy_step():
                time.sleep(0.01)  # Simulate work
                return "completed"
            
            result = optimizer.optimize_simulation_step(dummy_step)
            assert result == "completed"
            
            # Get performance report
            report = optimizer.get_performance_report()
            assert "metrics" in report
            assert "optimizations_enabled" in report
            
        finally:
            optimizer.shutdown()


class TestAutoScaler:
    """Test auto-scaling functionality."""
    
    def test_scaling_policy_creation(self):
        """Test creation of scaling policies."""
        policy = ScalingPolicy(
            name="test_policy",
            cpu_scale_up_threshold=0.8,
            cpu_scale_down_threshold=0.3,
            min_instances=1,
            max_instances=5
        )
        
        assert policy.name == "test_policy"
        assert policy.cpu_scale_up_threshold == 0.8
        assert policy.min_instances == 1
    
    def test_scaling_engine_evaluation(self):
        """Test scaling decision evaluation."""
        policy = DEFAULT_POLICIES["conservative"]
        engine = ScalingEngine(policy)
        
        # Simulate high CPU usage
        engine.metrics_collector.agent_count = 100
        
        # Force high metrics to trigger scale-up
        import psutil
        original_cpu_percent = psutil.cpu_percent
        
        def mock_high_cpu(interval=None):
            return 85.0  # High CPU usage
        
        psutil.cpu_percent = mock_high_cpu
        
        try:
            decision = engine.evaluate_scaling()
            # Note: Due to consecutive_periods_required, first evaluation might return None
            # This is expected behavior
            
        finally:
            psutil.cpu_percent = original_cpu_percent
        
        # Test scaling status
        status = engine.get_scaling_status()
        assert "current_instances" in status
        assert "metrics" in status
        assert "thresholds" in status
    
    def test_auto_scaler_registration(self):
        """Test auto-scaler policy registration."""
        scaler = AutoScaler()
        
        policy = ScalingPolicy(name="test", min_instances=1, max_instances=3)
        
        def scaling_callback(decision):
            pass
        
        scaler.register_policy("test", policy, scaling_callback)
        
        engine = scaler.get_engine("test")
        assert engine is not None
        assert engine.policy.name == "test"
        
        # Test status retrieval
        status = scaler.get_all_status()
        assert "test" in status


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_simulation_performance_small(self):
        """Test performance with small simulation."""
        config = SwarmConfig(
            num_agents=50,
            arena_size=(500, 500),
            episode_length=100,
            seed=42
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=50)
        
        start_time = time.time()
        results = arena.run(episodes=3, verbose=False)
        duration = time.time() - start_time
        
        # Performance assertions
        assert duration < 10.0  # Should complete in under 10 seconds
        assert results.total_steps > 0
        
        steps_per_second = results.total_steps / duration
        assert steps_per_second > 10  # At least 10 steps per second
    
    def test_simulation_performance_medium(self):
        """Test performance with medium simulation."""
        config = SwarmConfig(
            num_agents=200,
            arena_size=(1000, 1000),
            episode_length=200,
            seed=42
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=100)
        arena.add_agents(CooperativeAgent, count=100)
        
        start_time = time.time()
        results = arena.run(episodes=2, verbose=False)
        duration = time.time() - start_time
        
        # Performance assertions
        assert duration < 30.0  # Should complete in under 30 seconds
        
        steps_per_second = results.total_steps / duration
        assert steps_per_second > 5  # At least 5 steps per second
    
    def test_concurrent_simulations(self):
        """Test running multiple simulations concurrently."""
        def run_simulation(sim_id):
            config = SwarmConfig(
                num_agents=30,
                arena_size=(400, 400),
                episode_length=50,
                seed=42 + sim_id
            )
            
            arena = Arena(config)
            arena.add_agents(Agent, count=30)
            
            start_time = time.time()
            results = arena.run(episodes=2, verbose=False)
            duration = time.time() - start_time
            
            return {
                'sim_id': sim_id,
                'duration': duration,
                'steps_per_second': results.total_steps / duration,
                'mean_reward': results.mean_reward
            }
        
        # Run 4 simulations concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_simulation, i) for i in range(4)]
            results = [future.result() for future in as_completed(futures)]
        
        assert len(results) == 4
        
        # All simulations should complete successfully
        for result in results:
            assert result['duration'] > 0
            assert result['steps_per_second'] > 0
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable over time."""
        config = SwarmConfig(
            num_agents=100,
            arena_size=(800, 800),
            episode_length=100,
            seed=42
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=100)
        
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss
        
        # Run multiple episodes
        for _ in range(5):
            arena.run(episodes=1, verbose=False)
            arena.reset()  # Reset between episodes
        
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / initial_memory
        
        # Memory growth should be minimal (less than 50%)
        assert memory_growth < 0.5, f"Memory grew by {memory_growth:.1%}"
    
    def test_scaling_performance(self):
        """Test performance scaling with different agent counts."""
        agent_counts = [10, 50, 100, 200]
        performance_results = []
        
        for num_agents in agent_counts:
            config = SwarmConfig(
                num_agents=num_agents,
                arena_size=(800, 800),
                episode_length=50,
                seed=42
            )
            
            arena = Arena(config)
            arena.add_agents(Agent, count=num_agents)
            
            start_time = time.time()
            results = arena.run(episodes=1, verbose=False)
            duration = time.time() - start_time
            
            steps_per_second = results.total_steps / duration
            steps_per_agent_per_second = steps_per_second / num_agents
            
            performance_results.append({
                'agent_count': num_agents,
                'steps_per_second': steps_per_second,
                'steps_per_agent_per_second': steps_per_agent_per_second
            })
        
        # Performance should not degrade drastically with more agents
        # (allowing for some overhead, but should still be reasonable)
        first_perf = performance_results[0]['steps_per_agent_per_second']
        last_perf = performance_results[-1]['steps_per_agent_per_second']
        
        performance_ratio = last_perf / first_perf
        assert performance_ratio > 0.1, f"Performance degraded too much: {performance_ratio:.2f}"


class TestMemoryEfficiency:
    """Test memory efficiency and leak detection."""
    
    def test_no_memory_leaks_in_episodes(self):
        """Test that running episodes doesn't leak memory."""
        config = SwarmConfig(
            num_agents=50,
            arena_size=(500, 500),
            episode_length=100,
            seed=42
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=50)
        
        import gc
        import psutil
        
        process = psutil.Process()
        
        # Run first episode and measure memory
        arena.run(episodes=1, verbose=False)
        gc.collect()
        baseline_memory = process.memory_info().rss
        
        # Run several more episodes
        for _ in range(10):
            arena.reset()
            arena.run(episodes=1, verbose=False)
            gc.collect()
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - baseline_memory
        
        # Memory growth should be minimal (less than 10MB)
        assert memory_growth < 10 * 1024 * 1024, f"Memory leaked: {memory_growth / 1024 / 1024:.1f} MB"
    
    def test_agent_object_reuse(self):
        """Test that agent objects can be reused efficiently."""
        config = SwarmConfig(
            num_agents=100,
            arena_size=(500, 500),
            episode_length=50,
            seed=42
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=100)
        
        # Get initial agent IDs
        initial_agent_ids = set(arena.agents.keys())
        
        # Run and reset multiple times
        for _ in range(5):
            arena.run(episodes=1, verbose=False)
            arena.reset()
        
        # Agent IDs should remain the same (objects reused)
        final_agent_ids = set(arena.agents.keys())
        assert initial_agent_ids == final_agent_ids
    
    def test_large_simulation_memory_limit(self):
        """Test that large simulations don't exceed memory limits."""
        config = SwarmConfig(
            num_agents=500,
            arena_size=(2000, 2000),
            episode_length=200,
            seed=42
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=500)
        
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss
        
        # Run simulation
        arena.run(episodes=1, verbose=False)
        
        peak_memory = process.memory_info().rss
        memory_used = peak_memory - initial_memory
        
        # Should not use more than 500MB for this simulation
        max_memory_mb = 500
        memory_used_mb = memory_used / (1024 * 1024)
        
        assert memory_used_mb < max_memory_mb, f"Used {memory_used_mb:.1f}MB (limit: {max_memory_mb}MB)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])