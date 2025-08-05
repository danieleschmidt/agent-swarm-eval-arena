"""Advanced feature tests for Swarm Arena."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from swarm_arena import (
    Arena, SwarmConfig, LearningAgent, HierarchicalAgent, 
    SwarmAgent, AdaptiveAgent, TelemetryCollector
)
from swarm_arena.utils.spatial import SpatialHashGrid
from swarm_arena.monitoring.streaming import StreamingServer


class TestAdvancedAgentBehaviors:
    """Test sophisticated agent behaviors."""
    
    def test_learning_agent_q_learning(self):
        """Test Q-learning implementation in LearningAgent."""
        agent = LearningAgent(
            agent_id=1,
            initial_position=np.array([100, 100]),
            learning_rate=0.5,
            epsilon=0.1
        )
        
        # Test state discretization
        observation = {
            "position": [100, 100],
            "nearby_agents": [[120, 110], [90, 95]],
            "resources": [[150, 150]]
        }
        
        state_key = agent._get_state_key(observation)
        assert isinstance(state_key, str)
        assert "2_2_1" in state_key  # pos_discrete_nearby_resources
        
        # Test action selection
        action = agent.act(observation)
        assert 0 <= action <= 5
        
        # Test learning update
        next_observation = {
            "position": [105, 100],
            "nearby_agents": [[120, 110]],
            "resources": []
        }
        
        agent.learn(reward=1.0, next_observation=next_observation, done=False)
        
        # Should have updated Q-table
        assert len(agent.q_table) > 0
        assert len(agent.memory) > 0
        
        # Q-values should be updated
        if agent.last_state in agent.q_table:
            q_values = agent.q_table[agent.last_state]
            assert np.any(q_values != 0)  # Some Q-value should be non-zero
    
    def test_hierarchical_agent_strategy_switching(self):
        """Test hierarchical agent strategy selection."""
        agent = HierarchicalAgent(
            agent_id=2,
            initial_position=np.array([200, 200])
        )
        
        # Test strategy selection with many resources
        obs_harvest = {
            "position": [200, 200],
            "resources": [[190, 190], [210, 210], [180, 220], [220, 180]],
            "nearby_agents": []
        }
        
        strategy = agent._select_strategy(obs_harvest)
        assert strategy == "harvest"
        
        # Test strategy selection with many agents
        obs_cooperate = {
            "position": [200, 200],
            "resources": [],
            "nearby_agents": [[190, 190], [210, 210], [180, 220], [220, 180], [195, 205], [205, 195]]
        }
        
        strategy = agent._select_strategy(obs_cooperate)
        assert strategy == "cooperate"
        
        # Test strategy execution
        agent.strategy = "harvest"
        action = agent._execute_strategy(obs_harvest)
        assert 0 <= action <= 5
        
        # Test strategy timer
        initial_timer = agent.strategy_timer
        agent.act(obs_harvest)
        assert agent.strategy_timer == initial_timer + 1
    
    def test_swarm_agent_pattern_detection(self):
        """Test swarm pattern detection."""
        agent = SwarmAgent(
            agent_id=3,
            initial_position=np.array([300, 300])
        )
        
        # Test flocking detection
        obs_flocking = {
            "position": [300, 300],
            "nearby_agents": [
                [295, 295], [305, 305], [298, 302], [302, 298], 
                [297, 303], [303, 297]
            ],
            "resources": []
        }
        
        pattern_info = agent._detect_swarm_patterns(obs_flocking)
        assert pattern_info["pattern"] == "flocking"
        assert pattern_info["strength"] > 0.5
        assert "center" in pattern_info
        
        # Test spreading detection
        obs_spreading = {
            "position": [300, 300],
            "nearby_agents": [
                [200, 200], [400, 400], [250, 350], [350, 250]
            ],
            "resources": []
        }
        
        pattern_info = agent._detect_swarm_patterns(obs_spreading)
        assert pattern_info["pattern"] == "spreading"
        
        # Test isolated detection
        obs_isolated = {
            "position": [300, 300],
            "nearby_agents": [],
            "resources": []
        }
        
        pattern_info = agent._detect_swarm_patterns(obs_isolated)
        assert pattern_info["pattern"] == "isolated"
    
    def test_adaptive_agent_behavior_adaptation(self):
        """Test adaptive agent behavior weight adjustment."""
        agent = AdaptiveAgent(
            agent_id=4,
            initial_position=np.array([400, 400])
        )
        
        # Initial weights should be balanced
        initial_weights = agent.behavior_weights.copy()
        assert all(abs(w - 0.25) < 0.01 for w in initial_weights.values())
        
        # Simulate positive performance
        for _ in range(15):
            agent.update_behavior_weights(1.0)  # Good rewards
        
        # Should have adapted weights
        current_weights = agent.behavior_weights
        assert current_weights != initial_weights
        
        # Weights should still sum to 1
        weight_sum = sum(current_weights.values())
        assert abs(weight_sum - 1.0) < 0.01
        
        # Should have performance history
        assert len(agent.performance_history) > 0
        
        # Test poor performance adaptation
        agent_poor = AdaptiveAgent(agent_id=5, initial_position=np.array([500, 500]))
        for _ in range(15):
            agent_poor.update_behavior_weights(-1.0)  # Poor rewards
        
        # Should revert to balanced weights
        poor_weights = agent_poor.behavior_weights
        assert all(abs(w - 0.25) < 0.1 for w in poor_weights.values())


class TestSpatialIndexingAdvanced:
    """Advanced spatial indexing tests."""
    
    def test_spatial_hash_grid_collision_handling(self):
        """Test hash grid handles position collisions."""
        grid = SpatialHashGrid(arena_size=(1000, 1000), cell_size=50)
        
        # Add agents at same position
        same_pos = np.array([100, 100])
        for i in range(10):
            grid.update_agent(i, same_pos)
        
        # All should be in same cell
        neighbors = grid.query_neighbors(same_pos, 10.0)
        assert len(neighbors) == 9  # 10 agents - 1 excluded
        
        # Test moving agents apart
        for i in range(10):
            new_pos = same_pos + np.array([i*5, i*5])
            grid.update_agent(i, new_pos)
        
        # Should now be spread across cells
        stats = grid.get_statistics()
        assert stats["occupied_cells"] > 1
    
    def test_spatial_grid_boundary_conditions(self):
        """Test spatial grid at arena boundaries."""
        grid = SpatialHashGrid(arena_size=(200, 200), cell_size=50)
        
        # Test boundary positions
        boundary_positions = [
            np.array([0, 0]),      # corner
            np.array([200, 200]),  # opposite corner
            np.array([0, 100]),    # edge
            np.array([100, 0]),    # edge
            np.array([199.9, 199.9])  # near boundary
        ]
        
        for i, pos in enumerate(boundary_positions):
            grid.update_agent(i, pos)
        
        # All agents should be properly indexed
        assert grid.get_statistics()["agent_count"] == 5
        
        # Queries at boundaries should work
        for pos in boundary_positions:
            neighbors = grid.query_neighbors(pos, 50.0)
            assert isinstance(neighbors, list)
    
    def test_spatial_grid_performance_scaling(self):
        """Test spatial grid performance with increasing agents."""
        grid = SpatialHashGrid(arena_size=(1000, 1000), cell_size=100)
        
        agent_counts = [100, 500, 1000]
        query_times = []
        
        for count in agent_counts:
            # Clear and repopulate
            grid.clear()
            
            # Add agents randomly
            positions = []
            for i in range(count):
                pos = np.array([
                    np.random.uniform(0, 1000),
                    np.random.uniform(0, 1000)
                ])
                positions.append(pos)
                grid.update_agent(i, pos)
            
            # Time queries
            start_time = time.time()
            for i in range(min(50, count)):  # Sample queries
                grid.query_neighbors(positions[i], 100.0)
            query_time = (time.time() - start_time) / min(50, count)
            query_times.append(query_time)
        
        # Performance should not degrade significantly
        assert query_times[0] < 0.01  # Base case should be fast
        if len(query_times) > 1:
            # 10x agents should not be 10x slower (spatial indexing benefit)
            assert query_times[-1] < query_times[0] * 5


class TestTelemetryAdvanced:
    """Advanced telemetry and monitoring tests."""
    
    def test_telemetry_callback_system(self):
        """Test telemetry callback functionality."""
        telemetry = TelemetryCollector(max_history=100, auto_start=False)
        
        # Create callback functions
        callback_data = []
        
        def test_callback(data):
            callback_data.append(data)
        
        def failing_callback(data):
            raise ValueError("Test error")
        
        # Add callbacks
        telemetry.add_callback(test_callback)
        telemetry.add_callback(failing_callback)
        
        # Start collection
        telemetry.start_collection()
        
        # Update telemetry
        telemetry.update_telemetry(step=1, fps=60.0, active_agents=10)
        time.sleep(0.2)  # Allow callback processing
        
        # Should have called successful callback
        assert len(callback_data) > 0
        
        # Failing callback should be removed after failures
        telemetry.update_telemetry(step=2, fps=60.0, active_agents=10)
        telemetry.update_telemetry(step=3, fps=60.0, active_agents=10)
        telemetry.update_telemetry(step=4, fps=60.0, active_agents=10)
        time.sleep(0.3)
        
        # Should have fewer callbacks now (failing one removed)
        assert len(telemetry.callbacks) < 2
        
        telemetry.cleanup()
    
    def test_telemetry_memory_management(self):
        """Test telemetry automatic memory management."""
        telemetry = TelemetryCollector(
            max_history=50,
            memory_limit_mb=1.0,
            auto_start=False
        )
        
        # Add many custom metrics
        for i in range(100):
            large_metrics = {f"metric_{j}": f"large_data_string_{j}" * 10 
                            for j in range(30)}
            telemetry.update_telemetry(**large_metrics)
        
        # Should trigger memory cleanup
        latest_data = telemetry.get_latest_data()
        assert len(latest_data.custom_metrics) <= 20  # Should be limited
        
        # History should respect bounds
        assert len(telemetry.data_history) <= 50
        
        telemetry.cleanup()
    
    def test_telemetry_export_formats(self):
        """Test telemetry data export functionality."""
        telemetry = TelemetryCollector(max_history=10, auto_start=False)
        
        # Add sample data
        for i in range(5):
            telemetry.update_telemetry(
                step=i,
                fps=60.0 - i,
                active_agents=10 + i,
                custom_metric=f"value_{i}"
            )
            # Manually add to history for testing
            telemetry.data_history.append(telemetry.get_latest_data())
        
        # Test JSON export
        json_data = telemetry.export_data(format="json")
        assert json_data.startswith("[")
        assert "timestamp" in json_data
        assert "fps" in json_data
        
        # Test CSV export
        csv_data = telemetry.export_data(format="csv")
        lines = csv_data.split('\n')
        assert len(lines) > 1  # Header + data rows
        assert "timestamp,step,fps" in lines[0]  # Header
        
        # Test compressed export
        compressed_json = telemetry.export_data(format="json", compress=True)
        assert len(compressed_json) <= len(json_data)  # Should be smaller or equal
        
        telemetry.cleanup()


class TestErrorRecoveryAndRobustness:
    """Test system robustness and error recovery."""
    
    def test_agent_exception_isolation(self):
        """Test that agent exceptions don't crash simulation."""
        
        class FailingAgent(LearningAgent):
            def act(self, observation):
                if self.agent_id == 0:  # First agent always fails
                    raise ValueError("Intentional test failure")
                return super().act(observation)
        
        config = SwarmConfig(
            num_agents=5,
            arena_size=(300, 300),
            episode_length=10,
            seed=555
        )
        
        arena = Arena(config)
        
        # Add failing agent and normal agents
        failing_agent = FailingAgent(agent_id=0, initial_position=np.array([100, 100]))
        arena.add_agent(failing_agent)
        arena.add_agents(LearningAgent, count=4)
        
        # Simulation should continue despite failing agent
        arena.reset()
        for step in range(5):
            observations, rewards, done, info = arena.step()
            
            # Should have observations for working agents
            working_agents = [obs for obs in observations.values() if obs]
            assert len(working_agents) >= 4  # At least the working agents
            
            # Failing agent should get safe fallback action (0)
            if 0 in rewards:
                # Failing agent might still get processed with fallback
                pass
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        config = SwarmConfig(
            num_agents=50,
            arena_size=(500, 500),
            episode_length=200,
            seed=777
        )
        
        arena = Arena(config)
        arena.add_agents(SwarmAgent, count=50)
        
        # Create memory-intensive scenario
        telemetry = TelemetryCollector(max_history=1000, auto_start=True)
        
        try:
            # Run longer simulation
            results = arena.run(episodes=3, verbose=False)
            
            # Should complete without memory errors
            assert results.mean_reward is not None
            assert len(results.episode_rewards) == 50
            
            # Memory management should have occurred
            assert len(arena.step_times) <= arena._max_performance_history
            
        finally:
            telemetry.cleanup()
    
    def test_concurrent_access_safety(self):
        """Test thread safety of shared components."""
        import threading
        
        telemetry = TelemetryCollector(max_history=500, auto_start=True)
        
        def update_telemetry_worker(worker_id):
            for i in range(100):
                telemetry.update_telemetry(
                    step=i,
                    worker_id=worker_id,
                    fps=60.0 + worker_id,
                    active_agents=10
                )
                time.sleep(0.001)  # Small delay
        
        # Start multiple worker threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=update_telemetry_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have processed updates from all threads
        stats = telemetry.get_statistics()
        assert stats["update_count"] == 300  # 3 threads * 100 updates
        
        telemetry.cleanup()


class TestDistributedSystemIntegration:
    """Test distributed system components."""
    
    @pytest.mark.skipif(True, reason="Requires Ray cluster setup")
    def test_distributed_arena_setup(self):
        """Test distributed arena initialization."""
        # This would require Ray cluster setup
        # Placeholder for distributed testing
        pass
    
    def test_streaming_server_mock(self):
        """Test streaming server with mocked connections."""
        try:
            from swarm_arena.monitoring.streaming import StreamingServer
            
            server = StreamingServer(port=8080, auto_start=False)
            
            # Mock WebSocket connection
            mock_websocket = Mock()
            server.clients.add(mock_websocket)
            
            # Test message broadcasting
            test_data = {"step": 1, "fps": 60.0, "agents": 10}
            server.broadcast_data(test_data)
            
            # Should have attempted to send to mock client
            mock_websocket.send.assert_called()
            
        except ImportError:
            pytest.skip("Streaming server not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])