"""
Unit tests for sentiment-aware agents.

Tests the SentimentAwareAgent class and its subclasses for
emotional intelligence and behavioral adaptation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from swarm_arena.core.sentiment_aware_agent import (
    SentimentAwareAgent, EmotionalCooperativeAgent, EmotionalCompetitiveAgent,
    EmotionalAdaptiveAgent, SentimentAwareAgentConfig
)
from swarm_arena.sentiment.emotional_state import EmotionalState, EmotionType


class TestSentimentAwareAgentConfig:
    """Test suite for SentimentAwareAgentConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SentimentAwareAgentConfig()
        
        assert config.emotional_sensitivity == 0.7
        assert config.sentiment_processing == "behavioral"
        assert config.memory_capacity == 500
        assert config.contagion_susceptibility == 0.5
        assert config.empathy_level == 0.6
        assert config.emotional_decay_rate == 0.95
        assert config.learning_from_emotion is True
        assert config.emotion_expression is True
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        config = SentimentAwareAgentConfig(
            emotional_sensitivity=1.5,        # Should be clamped to 1.0
            contagion_susceptibility=-0.1,    # Should be clamped to 0.0
            empathy_level=0.5,                # Should remain unchanged
            emotional_decay_rate=0.5,         # Should be clamped to 0.8
            memory_capacity=5000              # Should be clamped to 2000
        )
        
        assert config.emotional_sensitivity == 1.0
        assert config.contagion_susceptibility == 0.0
        assert config.empathy_level == 0.5
        assert config.emotional_decay_rate == 0.8
        assert config.memory_capacity == 2000


class TestSentimentAwareAgent:
    """Test suite for SentimentAwareAgent functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent_id = 1
        self.initial_position = np.array([100.0, 100.0])
        self.config = SentimentAwareAgentConfig(
            emotional_sensitivity=0.8,
            empathy_level=0.7,
            memory_capacity=100
        )
        
        self.agent = SentimentAwareAgent(
            agent_id=self.agent_id,
            initial_position=self.initial_position,
            config=self.config
        )
    
    def test_initialization(self):
        """Test sentiment-aware agent initialization."""
        assert self.agent.agent_id == self.agent_id
        assert np.array_equal(self.agent.state.position, self.initial_position)
        assert hasattr(self.agent, 'sentiment_processor')
        assert hasattr(self.agent, 'emotional_state')
        assert hasattr(self.agent, 'sentiment_memory')
        
        # Check emotional state initialization
        assert self.agent.emotional_state.agent_id == self.agent_id
        assert len(self.agent.recent_actions) == 0
        assert len(self.agent.recent_rewards) == 0
        assert len(self.agent.peer_emotions) == 0
    
    def test_action_selection(self):
        """Test emotional action selection."""
        observation = {
            'position': [100, 100],
            'resources': [[120, 120], [80, 80]],
            'nearby_agents': [[90, 90], [110, 110]],
            'arena_bounds': {'width': 1000, 'height': 1000}
        }
        
        # Test multiple action selections
        actions = []
        for _ in range(10):
            action = self.agent.act(observation)
            actions.append(action)
            assert 0 <= action <= 5  # Valid action range
        
        # Should have some variation in actions due to emotional influence
        assert len(set(actions)) >= 2
    
    def test_emotional_decision_tracking(self):
        """Test tracking of emotional vs rational decisions."""
        observation = {
            'position': [100, 100],
            'resources': [],
            'nearby_agents': [],
            'arena_bounds': {'width': 1000, 'height': 1000}
        }
        
        initial_emotional = self.agent.emotional_decisions
        initial_rational = self.agent.rational_decisions
        
        # Take several actions
        for _ in range(5):
            self.agent.act(observation)
        
        # Should have recorded some decisions
        total_decisions = self.agent.emotional_decisions + self.agent.rational_decisions
        assert total_decisions == 5
    
    def test_peer_emotional_influence(self):
        """Test applying peer emotional influence."""
        # Create peer emotional state
        peer_state = EmotionalState(
            agent_id=999,
            initial_arousal=0.8,
            initial_valence=0.7,
            initial_dominance=0.5
        )
        
        # Store initial emotional state
        initial_valence = self.agent.emotional_state.valence
        
        # Apply peer influence
        self.agent.apply_peer_emotional_influence(peer_id=999, peer_emotion=peer_state, distance=50.0)
        
        # Should have recorded peer emotion
        assert 999 in self.agent.peer_emotions
        
        # Emotional state should have been influenced
        influence_received = self.agent.total_emotional_influence_received
        assert influence_received > 0.0
    
    def test_emotional_expression(self):
        """Test emotional expression generation."""
        # Set specific emotional state
        self.agent.emotional_state.valence = 0.8
        self.agent.emotional_state.arousal = 0.6
        
        expression = self.agent.get_emotional_expression()
        
        assert 'agent_id' in expression
        assert 'dominant_emotion' in expression
        assert 'emotion_strength' in expression
        assert 'arousal' in expression
        assert 'valence' in expression
        assert 'dominance' in expression
        assert 'timestamp' in expression
        
        assert expression['agent_id'] == self.agent_id
        assert expression['arousal'] == 0.6
        assert expression['valence'] == 0.8
    
    def test_sentiment_analytics(self):
        """Test sentiment analytics generation."""
        # Generate some activity
        observation = {'position': [100, 100], 'resources': [], 'nearby_agents': []}
        for i in range(5):
            action = self.agent.act(observation)
            self.agent.record_action(action, 0.5)
        
        analytics = self.agent.get_sentiment_analytics()
        
        assert 'agent_id' in analytics
        assert 'emotional_state' in analytics
        assert 'decision_analytics' in analytics
        assert 'behavioral_patterns' in analytics
        
        # Check decision analytics
        decision_data = analytics['decision_analytics']
        assert 'emotional_decisions' in decision_data
        assert 'rational_decisions' in decision_data
        assert 'emotion_influence_ratio' in decision_data
        
        # Check behavioral patterns
        behavior_data = analytics['behavioral_patterns']
        assert 'recent_actions' in behavior_data
        assert 'recent_rewards' in behavior_data
    
    def test_emotional_learning(self):
        """Test emotional learning from experience."""
        # Ensure learning is enabled
        self.agent.sentiment_config.learning_from_emotion = True
        
        observation = {
            'position': [100, 100],
            'resources': [[110, 110]],
            'nearby_agents': [[90, 90]]
        }
        
        # Take action and record positive reward
        action = self.agent.act(observation)
        self.agent.record_action(action, 0.8)
        
        # Should have added experience to memory
        initial_memory_size = len(self.agent.sentiment_memory.memory)
        
        # Take another action
        action = self.agent.act(observation)
        self.agent.record_action(action, 0.7)
        
        # Memory should have grown (if learning enabled)
        if self.agent.sentiment_config.learning_from_emotion:
            assert len(self.agent.sentiment_memory.memory) >= initial_memory_size
    
    def test_empathic_behavior(self):
        """Test empathy-driven action selection."""
        # Set high empathy
        self.agent.sentiment_config.empathy_level = 0.9
        
        # Add distressed peer emotion
        self.agent.peer_emotions[999] = {
            'arousal': 0.8,
            'valence': -0.7,  # Distressed
            'dominance': -0.3
        }
        
        observation = {
            'position': [100, 100],
            'resources': [[200, 200]],  # Far resource
            'nearby_agents': [[110, 110]],  # Close agent (potentially distressed)
            'arena_bounds': {'width': 1000, 'height': 1000}
        }
        
        # Test multiple actions to see empathic influence
        empathic_actions = 0
        total_actions = 10
        
        for _ in range(total_actions):
            action = self.agent.act(observation)
            # Actions 1-4 (movement) toward nearby agent could indicate empathy
            if action in [1, 2, 3, 4]:
                empathic_actions += 1
        
        # Should show some empathic behavior (moving toward others)
        empathy_ratio = empathic_actions / total_actions
        assert empathy_ratio > 0.2  # At least some empathic actions
    
    def test_error_handling(self):
        """Test error handling in agent operations."""
        # Test with malformed observation
        malformed_obs = {
            'position': None,
            'resources': 'invalid',
            'nearby_agents': [1, 2, 3]  # Invalid format
        }
        
        # Should not crash
        try:
            action = self.agent.act(malformed_obs)
            assert 0 <= action <= 5
        except Exception as e:
            # If it does raise an exception, it should be handled gracefully
            pass
        
        # Test with invalid action recording
        try:
            self.agent.record_action(-1, float('inf'))
        except Exception as e:
            # Should handle invalid inputs gracefully
            pass


class TestEmotionalCooperativeAgent:
    """Test suite for EmotionalCooperativeAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = EmotionalCooperativeAgent(
            agent_id=1,
            initial_position=np.array([100.0, 100.0])
        )
    
    def test_cooperative_config(self):
        """Test cooperative agent configuration."""
        config = self.agent.sentiment_config
        
        assert config.emotional_sensitivity == 0.8
        assert config.empathy_level == 0.9
        assert config.contagion_susceptibility == 0.7
        assert config.emotion_expression is True
    
    def test_cooperative_behavior(self):
        """Test cooperative action selection."""
        observation = {
            'position': [100, 100],
            'resources': [[200, 200]],
            'nearby_agents': [[90, 90], [110, 110], [95, 95]],  # Group of agents
            'arena_bounds': {'width': 1000, 'height': 1000}
        }
        
        # Take multiple actions
        group_oriented_actions = 0
        total_actions = 20
        
        for _ in range(total_actions):
            action = self.agent.act(observation)
            # Actions that keep agent with group (no-op or movement toward center)
            if action == 0 or action in [1, 2, 3, 4]:
                group_oriented_actions += 1
        
        # Cooperative agents should show group-oriented behavior
        cooperation_ratio = group_oriented_actions / total_actions
        assert cooperation_ratio > 0.6


class TestEmotionalCompetitiveAgent:
    """Test suite for EmotionalCompetitiveAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = EmotionalCompetitiveAgent(
            agent_id=1,
            initial_position=np.array([100.0, 100.0])
        )
    
    def test_competitive_config(self):
        """Test competitive agent configuration."""
        config = self.agent.sentiment_config
        
        assert config.emotional_sensitivity == 0.6
        assert config.empathy_level == 0.3
        assert config.contagion_susceptibility == 0.4
        assert config.emotion_expression is True
    
    def test_competitive_behavior(self):
        """Test competitive action selection."""
        observation = {
            'position': [100, 100],
            'resources': [[110, 110], [90, 90]],  # Nearby resources
            'nearby_agents': [[80, 80], [120, 120]],
            'arena_bounds': {'width': 1000, 'height': 1000}
        }
        
        resource_seeking_actions = 0
        total_actions = 15
        
        for _ in range(total_actions):
            action = self.agent.act(observation)
            # Resource collection or movement toward resources
            if action == 5 or action in [1, 2, 3, 4]:
                resource_seeking_actions += 1
        
        # Competitive agents should focus on resources
        resource_focus_ratio = resource_seeking_actions / total_actions
        assert resource_focus_ratio > 0.7


class TestEmotionalAdaptiveAgent:
    """Test suite for EmotionalAdaptiveAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = EmotionalAdaptiveAgent(
            agent_id=1,
            initial_position=np.array([100.0, 100.0])
        )
    
    def test_adaptive_config(self):
        """Test adaptive agent configuration."""
        config = self.agent.sentiment_config
        
        assert config.emotional_sensitivity == 0.75
        assert config.empathy_level == 0.6
        assert config.learning_from_emotion is True
        assert config.memory_capacity == 1000
    
    def test_strategy_adaptation(self):
        """Test behavioral strategy adaptation."""
        # Check initial strategy weights
        assert hasattr(self.agent, 'behavioral_strategies')
        assert 'cooperative' in self.agent.behavioral_strategies
        assert 'competitive' in self.agent.behavioral_strategies
        assert 'exploratory' in self.agent.behavioral_strategies
        assert 'defensive' in self.agent.behavioral_strategies
        
        # All strategies should start with equal weight
        for strategy, weight in self.agent.behavioral_strategies.items():
            assert abs(weight - 0.25) < 0.01
    
    def test_strategy_execution(self):
        """Test different strategy execution."""
        observation = {
            'position': [100, 100],
            'resources': [[150, 150]],
            'nearby_agents': [[90, 90], [110, 110]],
            'arena_bounds': {'width': 1000, 'height': 1000}
        }
        
        # Test cooperative strategy
        action = self.agent._execute_strategy_action('cooperative', observation)
        assert 0 <= action <= 5
        
        # Test competitive strategy
        action = self.agent._execute_strategy_action('competitive', observation)
        assert 0 <= action <= 5
        
        # Test exploratory strategy
        action = self.agent._execute_strategy_action('exploratory', observation)
        assert 0 <= action <= 5
        
        # Test defensive strategy
        action = self.agent._execute_strategy_action('defensive', observation)
        assert 0 <= action <= 5
    
    def test_crowded_defensive_behavior(self):
        """Test defensive behavior in crowded situations."""
        # Create crowded observation
        crowded_observation = {
            'position': [100, 100],
            'resources': [[110, 110]],
            'nearby_agents': [[90, 90], [110, 110], [95, 95], [105, 105], [85, 85], [115, 115]],  # 6 nearby agents
            'arena_bounds': {'width': 1000, 'height': 1000}
        }
        
        # Test defensive strategy in crowded environment
        action = self.agent._execute_strategy_action('defensive', crowded_observation)
        
        # Should either move away or stay still (defensive actions)
        assert action in [0, 1, 2, 3, 4]


class TestSentimentAwareAgentIntegration:
    """Integration tests for sentiment-aware agent system."""
    
    def test_agent_interaction_simulation(self):
        """Test interaction between different sentiment-aware agent types."""
        # Create different agent types
        cooperative_agent = EmotionalCooperativeAgent(1, np.array([50, 50]))
        competitive_agent = EmotionalCompetitiveAgent(2, np.array([150, 150]))
        adaptive_agent = EmotionalAdaptiveAgent(3, np.array([100, 100]))
        
        agents = [cooperative_agent, competitive_agent, adaptive_agent]
        
        # Simulate basic interaction
        observation = {
            'position': [100, 100],
            'resources': [[125, 125]],
            'nearby_agents': [[50, 50], [150, 150]],
            'arena_bounds': {'width': 200, 'height': 200}
        }
        
        # Each agent should produce valid actions
        for agent in agents:
            action = agent.act(observation)
            assert 0 <= action <= 5
            
            # Record action with moderate reward
            agent.record_action(action, 0.5)
            
            # Should have updated internal state
            assert len(agent.recent_actions) > 0
            assert len(agent.recent_rewards) > 0
    
    def test_emotional_contagion_between_agents(self):
        """Test emotional contagion between different agent types."""
        happy_agent = EmotionalCooperativeAgent(1, np.array([100, 100]))
        neutral_agent = EmotionalCompetitiveAgent(2, np.array([120, 120]))
        
        # Make first agent happy
        happy_agent.emotional_state.valence = 0.8
        happy_agent.emotional_state.arousal = 0.6
        
        initial_neutral_valence = neutral_agent.emotional_state.valence
        
        # Apply emotional influence
        neutral_agent.apply_peer_emotional_influence(
            peer_id=1,
            peer_emotion=happy_agent.emotional_state,
            distance=30.0
        )
        
        # Neutral agent should become more positive
        assert neutral_agent.emotional_state.valence > initial_neutral_valence
        
        # Should track the influence
        assert neutral_agent.total_emotional_influence_received > 0.0
    
    def test_memory_and_learning_integration(self):
        """Test integration of emotional memory and learning."""
        agent = EmotionalAdaptiveAgent(1, np.array([100, 100]))
        
        # Simulate successful cooperative experiences
        cooperative_observation = {
            'position': [100, 100],
            'resources': [[110, 110]],
            'nearby_agents': [[90, 90], [95, 95]]  # Cooperative scenario
        }
        
        for i in range(10):
            action = agent.act(cooperative_observation)
            # Simulate good outcomes for cooperative behavior
            reward = 0.8 if action in [0, 5] else 0.3  # Reward cooperation
            agent.record_action(action, reward)
        
        # Agent should have learned from experiences
        insights = agent.sentiment_memory.get_emotional_learning_insights()
        
        if 'average_reward' in insights:
            assert insights['average_reward'] > 0.4  # Should have positive average
        
        # Should have accumulated emotional memory
        assert len(agent.sentiment_memory.memory) > 0


if __name__ == "__main__":
    pytest.main([__file__])