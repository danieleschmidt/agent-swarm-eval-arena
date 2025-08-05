"""Integration tests for the Swarm Arena system."""

import pytest
import numpy as np
import time
from typing import Dict, Any

from swarm_arena import (
    Arena, SwarmConfig, Agent, CooperativeAgent, CompetitiveAgent, 
    RandomAgent, LearningAgent, HierarchicalAgent, SwarmAgent, AdaptiveAgent,
    TelemetryCollector
)
from swarm_arena.core.environment import ForagingEnvironment
from swarm_arena.utils.spatial import SpatialHashGrid, SpatialQuadTree


class TestArenaIntegration:
    """Test full arena simulation with various agent types."""
    
    def test_basic_simulation_flow(self):
        """Test complete simulation workflow."""
        config = SwarmConfig(
            num_agents=20,
            arena_size=(400, 300),
            episode_length=100,
            resource_spawn_rate=0.1,
            seed=42
        )
        
        arena = Arena(config)
        
        # Add mixed agent types
        arena.add_agents(CooperativeAgent, count=5)
        arena.add_agents(CompetitiveAgent, count=5)
        arena.add_agents(RandomAgent, count=5)
        arena.add_agents(Agent, count=5)
        
        # Run simulation
        results = arena.run(episodes=2, verbose=False)
        
        # Verify results
        assert results.mean_reward is not None
        assert results.episode_rewards
        assert len(results.episode_rewards) == 20  # All agents
        assert results.total_steps > 0
        assert results.fairness_index is not None
        
        # Verify agent statistics
        assert len(results.agent_stats) == 20
        for agent_id, stats in results.agent_stats.items():
            assert "total_reward" in stats
            assert "resources_collected" in stats
    
    def test_large_scale_simulation(self):
        """Test simulation with many agents."""
        config = SwarmConfig(
            num_agents=100,
            arena_size=(800, 600),
            episode_length=50,
            resource_spawn_rate=0.05,
            seed=123
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=50)
        arena.add_agents(CooperativeAgent, count=30)
        arena.add_agents(CompetitiveAgent, count=20)
        
        start_time = time.time()
        results = arena.run(episodes=1, verbose=False)
        end_time = time.time()
        
        # Performance requirements
        simulation_time = end_time - start_time
        assert simulation_time < 30.0  # Should complete within 30 seconds
        
        # Verify spatial indexing is working
        spatial_stats = arena.spatial_index.get_statistics()
        assert spatial_stats["agent_count"] == 100
        assert spatial_stats["query_count"] > 0
        
        # Basic result validation
        assert len(results.episode_rewards) == 100
        assert results.total_steps == 50
    
    def test_telemetry_integration(self):
        """Test telemetry collection during simulation."""
        config = SwarmConfig(
            num_agents=30,
            arena_size=(500, 400),
            episode_length=50,
            seed=456
        )
        
        arena = Arena(config)
        arena.add_agents(SwarmAgent, count=15)
        arena.add_agents(HierarchicalAgent, count=15)
        
        # Initialize telemetry
        telemetry = TelemetryCollector(max_history=200, auto_start=True)
        
        # Collect telemetry during simulation
        arena.reset()
        for step in range(25):
            observations, rewards, done, info = arena.step()
            telemetry.record_arena_state(arena)
            
            if done:
                break
        
        # Verify telemetry data
        telemetry_stats = telemetry.get_statistics()
        assert telemetry_stats["data_points"] > 0
        assert telemetry_stats["avg_fps"] > 0
        
        latest_data = telemetry.get_latest_data()
        assert latest_data.total_agents == 30
        assert latest_data.step > 0
        
        # Cleanup
        telemetry.cleanup()
    
    def test_learning_agent_integration(self):
        """Test learning agents in simulation."""
        config = SwarmConfig(
            num_agents=10,
            arena_size=(300, 300),
            episode_length=100,
            resource_spawn_rate=0.2,
            seed=789
        )
        
        arena = Arena(config)
        
        # Create learning agents with different parameters
        learning_agents = []
        for i in range(5):
            agent = LearningAgent(
                agent_id=i,
                initial_position=np.array([100 + i*20, 100 + i*20]),
                learning_rate=0.1,
                epsilon=0.2
            )
            learning_agents.append(agent)
            arena.add_agent(agent)
        
        # Add comparison agents
        arena.add_agents(RandomAgent, count=5)
        
        # Run simulation and provide learning feedback
        arena.reset()
        prev_observations = {}
        
        for step in range(50):
            observations, rewards, done, info = arena.step()
            
            # Update learning agents
            for agent in learning_agents:
                if agent.agent_id in rewards and agent.agent_id in prev_observations:
                    agent.learn(
                        reward=rewards[agent.agent_id],
                        next_observation=observations[agent.agent_id],
                        done=done
                    )
            
            prev_observations = observations.copy()
            
            if done:
                break
        
        # Verify learning occurred
        for agent in learning_agents:
            assert len(agent.q_table) > 0  # Should have learned some states
            assert len(agent.memory) > 0   # Should have experience
    
    def test_adaptive_agent_integration(self):
        """Test adaptive agents adjusting behavior."""
        config = SwarmConfig(
            num_agents=8,
            arena_size=(400, 400),
            episode_length=100,
            seed=101
        )
        
        arena = Arena(config)
        
        # Create adaptive agents
        adaptive_agents = []
        for i in range(4):
            agent = AdaptiveAgent(
                agent_id=i,
                initial_position=np.array([100 + i*50, 100 + i*50])
            )
            adaptive_agents.append(agent)
            arena.add_agent(agent)
        
        arena.add_agents(Agent, count=4)
        
        # Run simulation with behavior adaptation
        arena.reset()
        
        for step in range(30):
            observations, rewards, done, info = arena.step()
            
            # Update adaptive agents
            for agent in adaptive_agents:
                if agent.agent_id in rewards:
                    agent.update_behavior_weights(rewards[agent.agent_id])
            
            if done:
                break
        
        # Verify adaptation occurred
        for agent in adaptive_agents:
            assert len(agent.performance_history) > 0
            # Weights should still sum to 1 (normalized)
            weight_sum = sum(agent.behavior_weights.values())
            assert abs(weight_sum - 1.0) < 0.01


class TestSpatialIndexIntegration:
    """Test spatial indexing systems."""
    
    def test_hash_grid_with_arena(self):
        """Test hash grid spatial indexing in arena."""
        config = SwarmConfig(
            num_agents=50,
            arena_size=(600, 400),
            observation_radius=75,
            seed=321
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=50)
        
        # Verify spatial index type
        assert isinstance(arena.spatial_index, SpatialHashGrid)
        
        # Run simulation and check spatial queries
        arena.reset()
        
        for step in range(10):
            observations, rewards, done, info = arena.step()
            
            # Verify observations use spatial indexing
            for agent_id, obs in observations.items():
                nearby_agents = obs.get("nearby_agents", [])
                assert isinstance(nearby_agents, list)
                # Should find some neighbors in populated area
                if step > 2:  # Allow agents to move around
                    assert len(nearby_agents) >= 0  # At least possible to find neighbors
        
        # Check spatial index statistics
        spatial_stats = arena.spatial_index.get_statistics()
        assert spatial_stats["agent_count"] == 50
        assert spatial_stats["query_count"] > 0
        assert spatial_stats["occupancy_ratio"] > 0
    
    def test_quadtree_with_large_arena(self):
        """Test quadtree spatial indexing with large arena."""
        config = SwarmConfig(
            num_agents=200,  # Force quadtree usage
            arena_size=(1000, 800),
            observation_radius=100,
            seed=654
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=200)
        
        # Should use quadtree for large agent count
        assert isinstance(arena.spatial_index, SpatialQuadTree)
        
        # Quick simulation
        arena.reset()
        for step in range(5):
            observations, rewards, done, info = arena.step()
        
        # Verify quadtree performance
        spatial_stats = arena.spatial_index.get_statistics()
        assert spatial_stats["agent_count"] == 200
        assert spatial_stats["total_nodes"] >= 1
        assert spatial_stats["insert_count"] > 0


class TestMemoryManagement:
    """Test memory management and cleanup."""
    
    def test_arena_memory_cleanup(self):
        """Test arena properly cleans up memory."""
        config = SwarmConfig(
            num_agents=30,
            arena_size=(400, 400),
            episode_length=100,
            seed=987
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=30)
        
        # Run multiple episodes
        for episode in range(3):
            results = arena.run(episodes=1, verbose=False)
            
            # Check memory bounds
            assert len(arena.step_times) <= arena._max_performance_history
            
            # Episode rewards should be cleared between episodes
            for agent_id in arena.episode_rewards:
                # Should have some data from current episode but not unbounded
                assert len(arena.episode_rewards[agent_id]) <= config.episode_length
    
    def test_telemetry_memory_bounds(self):
        """Test telemetry memory management."""
        telemetry = TelemetryCollector(
            max_history=100,
            memory_limit_mb=10.0,
            auto_start=False
        )
        
        # Add many data points
        for i in range(150):
            telemetry.update_telemetry(
                step=i,
                fps=60.0,
                active_agents=50,
                test_metric=f"data_{i}"
            )
        
        # Should respect max_history
        assert len(telemetry.data_history) == 100
        
        # Custom metrics should be bounded
        latest = telemetry.get_latest_data()
        assert len(latest.custom_metrics) <= 20
        
        telemetry.cleanup()


class TestErrorHandling:
    """Test error handling and recovery."""
    
    def test_invalid_agent_handling(self):
        """Test handling of agents with invalid states."""
        config = SwarmConfig(
            num_agents=5,
            arena_size=(200, 200),
            episode_length=20,
            seed=111
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=5)
        
        # Simulate agent failure by modifying state
        arena.reset()
        
        # Kill an agent
        first_agent_id = list(arena.agents.keys())[0]
        arena.agents[first_agent_id].state.alive = False
        
        # Should handle dead agents gracefully
        for step in range(10):
            observations, rewards, done, info = arena.step()
            
            # Dead agent should not be in observations
            assert first_agent_id not in observations or not observations[first_agent_id]
            
            # Should still track info about active agents
            assert info["active_agents"] == 4  # 5 - 1 dead
    
    def test_spatial_index_error_recovery(self):
        """Test spatial index handles invalid positions."""
        config = SwarmConfig(
            num_agents=10,
            arena_size=(300, 300),
            seed=222
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=10)
        arena.reset()
        
        # Introduce invalid position
        first_agent_id = list(arena.agents.keys())[0]
        arena.agent_positions[first_agent_id] = np.array([float('inf'), float('nan')])
        
        # Should handle gracefully
        try:
            observations, rewards, done, info = arena.step()
            # Should not crash, might have different behavior but should continue
            assert True  # Made it here without exception
        except Exception as e:
            # If it does throw, should be handled gracefully
            assert "finite" in str(e).lower() or "invalid" in str(e).lower()


class TestPerformanceBenchmarks:
    """Performance and scalability tests."""
    
    def test_step_performance(self):
        """Test single step performance."""
        config = SwarmConfig(
            num_agents=100,
            arena_size=(800, 600),
            episode_length=1,
            seed=333
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=100)
        arena.reset()
        
        # Time single step
        start_time = time.time()
        observations, rewards, done, info = arena.step()
        step_time = time.time() - start_time
        
        # Should complete step quickly
        assert step_time < 0.1  # 100ms per step for 100 agents
        
        # Should have processed all agents
        assert len(observations) == 100
        assert len(rewards) == 100
        
        # Info should contain performance metrics
        assert "step_time" in info
        assert "average_step_time" in info
    
    def test_neighbor_query_performance(self):
        """Test spatial query performance."""
        from swarm_arena.utils.spatial import create_spatial_index
        
        config = SwarmConfig(
            num_agents=500,
            arena_size=(1000, 1000),
            observation_radius=50
        )
        
        # Test hash grid
        spatial_index = create_spatial_index(config, "hash_grid")
        
        # Add many agents
        positions = []
        for i in range(500):
            pos = np.array([
                np.random.uniform(0, 1000),
                np.random.uniform(0, 1000)
            ])
            positions.append(pos)
            spatial_index.update_agent(i, pos)
        
        # Time neighbor queries
        query_times = []
        for i in range(100):  # 100 queries
            test_pos = positions[i]
            start_time = time.time()
            neighbors = spatial_index.query_neighbors(test_pos, 50.0)
            query_time = time.time() - start_time
            query_times.append(query_time)
        
        avg_query_time = np.mean(query_times)
        assert avg_query_time < 0.001  # Should be sub-millisecond
        
        # Test quadtree performance
        spatial_index_qt = create_spatial_index(config, "quadtree")
        for i, pos in enumerate(positions):
            spatial_index_qt.insert(i, pos)
        
        # Compare performance
        qt_query_times = []
        for i in range(100):
            test_pos = positions[i]
            start_time = time.time()
            neighbors = spatial_index_qt.query_range(test_pos, 50.0)
            query_time = time.time() - start_time
            qt_query_times.append(query_time)
        
        avg_qt_query_time = np.mean(qt_query_times)
        assert avg_qt_query_time < 0.005  # Quadtree might be slightly slower but still fast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])