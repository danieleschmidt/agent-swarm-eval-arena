"""Tests for benchmark functionality."""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from swarm_arena.benchmarks.standard import StandardBenchmark
from swarm_arena.core.agent import RandomAgent, CooperativeAgent
from swarm_arena.core.config import SwarmConfig


class TestStandardBenchmark:
    """Test standard benchmark functionality."""
    
    def test_benchmark_initialization(self):
        """Test benchmark can be initialized."""
        benchmark = StandardBenchmark()
        assert benchmark is not None
    
    def test_single_benchmark_run(self):
        """Test running a single benchmark."""
        benchmark = StandardBenchmark()
        
        config = SwarmConfig(num_agents=5, episode_length=10, seed=42)
        
        result = benchmark.run_single_benchmark(
            RandomAgent,
            Mock,  # Mock environment
            config=config,
            num_episodes=1,
            seed=42
        )
        
        assert result is not None
        assert hasattr(result, 'agent_name')
        assert hasattr(result, 'mean_reward')
        assert hasattr(result, 'execution_time')
    
    def test_benchmark_with_failing_agent(self):
        """Test benchmark handles agent failures gracefully."""
        benchmark = StandardBenchmark()
        
        class FailingAgent:
            def act(self, observation):
                raise Exception("Intentional failure")
        
        config = SwarmConfig(num_agents=2, episode_length=5, seed=42)
        
        # Should handle failure gracefully
        result = benchmark.run_single_benchmark(
            FailingAgent,
            Mock,
            config=config,
            num_episodes=1,
            seed=42
        )
        
        # Should return a failed result with zero values
        assert result.agent_name == "FailingAgent"
        assert result.mean_reward == 0.0
        assert result.execution_time >= 0.0
    
    def test_multiple_agent_comparison(self):
        """Test comparing multiple agents."""
        benchmark = StandardBenchmark()
        
        config = SwarmConfig(num_agents=3, episode_length=5, seed=42)
        
        results = benchmark.run_all(
            agent_classes=[RandomAgent, CooperativeAgent],
            environments=[Mock],
            config=config,
            num_episodes=1
        )
        
        assert len(results) == 2  # Two agent types
        assert all(hasattr(r, 'mean_reward') for r in results)
    
    def test_benchmark_reproducibility(self):
        """Test that benchmarks are reproducible with same seed."""
        benchmark = StandardBenchmark()
        
        config = SwarmConfig(num_agents=2, episode_length=5, seed=42)
        
        result1 = benchmark.run_single_benchmark(
            RandomAgent,
            Mock,
            config=config,
            num_episodes=1,
            seed=42
        )
        
        result2 = benchmark.run_single_benchmark(
            RandomAgent,
            Mock,
            config=config,
            num_episodes=1,
            seed=42
        )
        
        # Results should be identical with same seed
        assert result1.mean_reward == result2.mean_reward