"""Core component tests for Swarm Arena."""

import pytest
import numpy as np
from swarm_arena import Arena, SwarmConfig, Agent
from swarm_arena.core.agent import CooperativeAgent, CompetitiveAgent, AgentState
from swarm_arena.core.environment import ForagingEnvironment, Resource


class TestSwarmConfig:
    """Test SwarmConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = SwarmConfig()
        
        assert config.num_agents == 100
        assert config.arena_size == (1000.0, 1000.0)
        assert config.episode_length == 1000
        assert config.physics_engine == "simple"
        assert config.seed is None
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = SwarmConfig(num_agents=50, arena_size=(500, 500))
        assert config.num_agents == 50
        
        # Invalid configs
        with pytest.raises(ValueError):
            SwarmConfig(num_agents=0)
        
        with pytest.raises(ValueError):
            SwarmConfig(arena_size=(0, 100))
        
        with pytest.raises(ValueError):
            SwarmConfig(episode_length=-1)
        
        with pytest.raises(ValueError):
            SwarmConfig(resource_spawn_rate=1.5)
    
    def test_config_serialization(self):
        """Test config to/from dict conversion."""
        config = SwarmConfig(num_agents=25, seed=42)
        config_dict = config.to_dict()
        
        assert config_dict["num_agents"] == 25
        assert config_dict["seed"] == 42
        
        restored_config = SwarmConfig.from_dict(config_dict)
        assert restored_config.num_agents == 25
        assert restored_config.seed == 42
    
    def test_config_copy(self):
        """Test config copying with overrides."""
        config = SwarmConfig(num_agents=100, seed=42)
        new_config = config.copy(num_agents=200, arena_size=(2000, 2000))
        
        assert new_config.num_agents == 200
        assert new_config.arena_size == (2000, 2000)
        assert new_config.seed == 42  # Unchanged


class TestAgentState:
    """Test AgentState class."""
    
    def test_agent_state_creation(self):
        """Test agent state initialization."""
        state = AgentState(
            agent_id=1,
            position=[100, 200],
            velocity=[5, -3]
        )
        
        assert state.agent_id == 1
        assert isinstance(state.position, np.ndarray)
        assert isinstance(state.velocity, np.ndarray)
        assert np.array_equal(state.position, [100, 200])
        assert np.array_equal(state.velocity, [5, -3])
        assert state.health == 1.0
        assert state.alive is True


class TestAgent:
    """Test Agent classes."""
    
    def test_base_agent_creation(self):
        """Test base agent initialization."""
        agent = Agent(agent_id=1, initial_position=[100, 200])
        
        assert agent.agent_id == 1
        assert np.array_equal(agent.state.position, [100, 200])
        assert np.array_equal(agent.state.velocity, [0, 0])
        assert len(agent.action_history) == 0
        assert len(agent.reward_history) == 0
    
    def test_agent_reset(self):
        """Test agent reset functionality."""
        agent = Agent(agent_id=1, initial_position=[100, 200])
        
        # Modify agent state
        agent.state.resources_collected = 5
        agent.action_history = [1, 2, 3]
        agent.reward_history = [0.1, 0.2, 0.3]
        
        # Reset agent
        agent.reset([300, 400])
        
        assert np.array_equal(agent.state.position, [300, 400])
        assert agent.state.resources_collected == 0
        assert len(agent.action_history) == 0
        assert len(agent.reward_history) == 0
    
    def test_agent_action_recording(self):
        """Test action and reward recording."""
        agent = Agent(agent_id=1, initial_position=[0, 0])
        
        agent.record_action(2, 0.5)
        agent.record_action(4, -0.1)
        
        assert agent.action_history == [2, 4]
        assert agent.reward_history == [0.5, -0.1]
        assert agent.state.last_action == 4
    
    def test_agent_stats(self):
        """Test agent statistics generation."""
        agent = Agent(agent_id=1, initial_position=[0, 0])
        
        agent.record_action(1, 1.0)
        agent.record_action(2, 0.5)
        agent.state.resources_collected = 3
        
        stats = agent.get_stats()
        
        assert stats["agent_id"] == 1
        assert stats["total_reward"] == 1.5
        assert stats["average_reward"] == 0.75
        assert stats["resources_collected"] == 3
        assert stats["actions_taken"] == 2
    
    def test_cooperative_agent(self):
        """Test cooperative agent behavior."""
        agent = CooperativeAgent(agent_id=1, initial_position=[100, 100])
        
        # Test with nearby agents
        observation = {
            "position": [100, 100],
            "nearby_agents": [[90, 90], [110, 110], [95, 105]],
            "resources": [],
            "arena_bounds": {"width": 1000, "height": 1000}
        }
        
        action = agent.act(observation)
        assert action in range(6)  # Valid action
    
    def test_competitive_agent(self):
        """Test competitive agent behavior."""
        agent = CompetitiveAgent(agent_id=1, initial_position=[100, 100])
        
        # Test with resources
        observation = {
            "position": [100, 100],
            "nearby_agents": [],
            "resources": [[150, 150], [80, 120]],
            "arena_bounds": {"width": 1000, "height": 1000}
        }
        
        action = agent.act(observation)
        assert action in [1, 2, 3, 4]  # Movement action expected


class TestEnvironment:
    """Test Environment classes."""
    
    def test_resource_creation(self):
        """Test resource object creation."""
        resource = Resource(position=[100, 200], value=1.5)
        
        assert isinstance(resource.position, np.ndarray)
        assert np.array_equal(resource.position, [100, 200])
        assert resource.value == 1.5
        assert resource.collected is False
    
    def test_foraging_environment(self):
        """Test foraging environment."""
        config = SwarmConfig(num_agents=10, seed=42)
        env = ForagingEnvironment(config)
        
        # Test reset
        env_state = env.reset()
        assert "environment" in env_state
        assert "bounds" in env_state
        assert len(env.state.resources) >= 2  # At least 2 initial resources
        
        # Test step
        actions = {0: 1, 1: 2, 2: 5}  # Some movement and collection
        obs, rewards, done, info = env.step(actions)
        
        assert len(rewards) == 3
        assert isinstance(done, bool)
        assert "collections" in info
    
    def test_environment_observation(self):
        """Test environment observation generation."""
        config = SwarmConfig(observation_radius=50, seed=42)
        env = ForagingEnvironment(config)
        env.reset()
        
        agent_pos = np.array([100, 100])
        obs = env.get_observation_for_agent(1, agent_pos)
        
        assert "position" in obs
        assert "nearby_resources" in obs
        assert "arena_bounds" in obs
        assert obs["position"] == [100, 100]


class TestArena:
    """Test Arena class."""
    
    def test_arena_creation(self):
        """Test arena initialization."""
        config = SwarmConfig(num_agents=10, seed=42)
        arena = Arena(config)
        
        assert arena.config == config
        assert len(arena.agents) == 0
        assert arena.current_step == 0
    
    def test_add_agents(self):
        """Test adding agents to arena."""
        config = SwarmConfig(seed=42)
        arena = Arena(config)
        
        arena.add_agents(Agent, count=5)
        arena.add_agents(CooperativeAgent, count=3)
        
        assert len(arena.agents) == 8
        assert len(arena.agent_positions) == 8
        assert len(arena.agent_velocities) == 8
    
    def test_arena_reset(self):
        """Test arena reset functionality."""
        config = SwarmConfig(seed=42)
        arena = Arena(config)
        arena.add_agents(Agent, count=3)
        
        # Modify state
        arena.current_step = 100
        
        # Reset
        state = arena.reset()
        
        assert arena.current_step == 0
        assert "num_agents" in state
        assert state["num_agents"] == 3
    
    def test_arena_step(self):
        """Test single arena step execution."""
        config = SwarmConfig(episode_length=10, seed=42)
        arena = Arena(config)
        arena.add_agents(Agent, count=2)
        
        arena.reset()
        
        # Execute step
        obs, rewards, done, info = arena.step()
        
        assert len(obs) == 2  # Two agents
        assert len(rewards) == 2
        assert isinstance(done, bool)
        assert "step" in info
        assert info["step"] == 1
    
    def test_arena_run(self):
        """Test running multiple episodes."""
        config = SwarmConfig(episode_length=5, seed=42)
        arena = Arena(config)
        arena.add_agents(Agent, count=2)
        
        results = arena.run(episodes=2, verbose=False)
        
        assert isinstance(results.mean_reward, float)
        assert len(results.agent_stats) == 2
        assert results.total_steps == 10  # 2 episodes * 5 steps each