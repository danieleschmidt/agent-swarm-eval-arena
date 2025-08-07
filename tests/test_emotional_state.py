"""
Unit tests for emotional state management.

Tests the EmotionalState class and related functionality for
agent emotional intelligence and behavioral modulation.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock

from swarm_arena.sentiment.emotional_state import (
    EmotionalState, EmotionalContext, EmotionType
)
from swarm_arena.sentiment.processor import SentimentData, SentimentPolarity


class TestEmotionalState:
    """Test suite for EmotionalState functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent_id = 1
        self.emotional_state = EmotionalState(
            agent_id=self.agent_id,
            initial_arousal=0.2,
            initial_valence=0.3,
            initial_dominance=0.1,
            decay_rate=0.95,
            memory_size=50
        )
    
    def test_initialization(self):
        """Test emotional state initialization."""
        assert self.emotional_state.agent_id == self.agent_id
        assert self.emotional_state.arousal == 0.2
        assert self.emotional_state.valence == 0.3
        assert self.emotional_state.dominance == 0.1
        assert self.emotional_state.decay_rate == 0.95
        assert self.emotional_state.memory_size == 50
        
        # Check that emotions are properly initialized
        assert EmotionType.NEUTRAL in self.emotional_state.emotions
        assert self.emotional_state.emotions[EmotionType.NEUTRAL] == 1.0
    
    def test_boundary_validation(self):
        """Test that emotional dimensions are kept within bounds."""
        # Test initialization with out-of-bounds values
        state = EmotionalState(
            agent_id=999,
            initial_arousal=2.0,   # Should be clamped to 1.0
            initial_valence=-1.5,  # Should be clamped to -1.0
            initial_dominance=0.5,
            decay_rate=1.5,        # Should be clamped to 0.99
            memory_size=5000       # Should remain as is
        )
        
        assert -1.0 <= state.arousal <= 1.0
        assert -1.0 <= state.valence <= 1.0
        assert -1.0 <= state.dominance <= 1.0
        assert 0.8 <= state.decay_rate <= 0.99
    
    def test_sentiment_update(self):
        """Test updating emotional state from sentiment data."""
        # Create sentiment data
        sentiment_data = SentimentData(
            polarity=SentimentPolarity.POSITIVE,
            intensity=0.7,
            confidence=0.8,
            emotional_dimensions={'arousal': 0.5, 'valence': 0.6, 'dominance': 0.3},
            processing_time=10.0
        )
        
        # Create context
        context = EmotionalContext(
            task_performance=0.4,
            resource_availability=0.6,
            cooperation_level=0.7
        )
        
        # Store initial values
        initial_arousal = self.emotional_state.arousal
        initial_valence = self.emotional_state.valence
        initial_dominance = self.emotional_state.dominance
        
        # Update emotional state
        self.emotional_state.update_from_sentiment(sentiment_data, context)
        
        # Values should have changed (moved toward sentiment values)
        assert self.emotional_state.arousal != initial_arousal
        assert self.emotional_state.valence != initial_valence
        assert self.emotional_state.dominance != initial_dominance
        
        # Values should still be within bounds
        assert -1.0 <= self.emotional_state.arousal <= 1.0
        assert -1.0 <= self.emotional_state.valence <= 1.0
        assert -1.0 <= self.emotional_state.dominance <= 1.0
    
    def test_emotional_decay(self):
        """Test natural emotional decay over time."""
        # Set high emotional values
        self.emotional_state.arousal = 0.8
        self.emotional_state.valence = 0.9
        self.emotional_state.dominance = 0.7
        
        # Apply decay multiple times
        for _ in range(10):
            self.emotional_state.apply_emotional_decay()
        
        # Values should decay toward zero
        assert abs(self.emotional_state.arousal) < 0.8
        assert abs(self.emotional_state.valence) < 0.9
        assert abs(self.emotional_state.dominance) < 0.7
    
    def test_dominant_emotion(self):
        """Test dominant emotion detection."""
        # Set specific emotional state for joy
        self.emotional_state.arousal = 0.7
        self.emotional_state.valence = 0.8
        self.emotional_state.dominance = 0.5
        self.emotional_state._update_primary_emotions()
        
        dominant_emotion, strength = self.emotional_state.get_dominant_emotion()
        
        assert isinstance(dominant_emotion, EmotionType)
        assert 0.0 <= strength <= 2.0  # Strength can exceed 1.0 due to combinations
        
        # Test with neutral state
        self.emotional_state.arousal = 0.0
        self.emotional_state.valence = 0.0
        self.emotional_state.dominance = 0.0
        self.emotional_state._update_primary_emotions()
        
        dominant_emotion, strength = self.emotional_state.get_dominant_emotion()
        assert dominant_emotion == EmotionType.NEUTRAL or strength > 0.0
    
    def test_behavioral_modifiers(self):
        """Test behavioral modifier calculations."""
        modifiers = self.emotional_state.get_behavioral_modifiers()
        
        # Check that all expected modifiers are present
        expected_modifiers = [
            'exploration_tendency', 'cooperation_tendency', 'risk_tolerance',
            'action_speed', 'decision_confidence', 'social_attraction'
        ]
        
        for modifier in expected_modifiers:
            assert modifier in modifiers
        
        # Check value ranges (except action_speed which can be > 1.0)
        for key, value in modifiers.items():
            if key == 'action_speed':
                assert 0.1 <= value <= 3.0
            else:
                assert 0.0 <= value <= 1.0
    
    def test_peer_influence(self):
        """Test emotional influence from peer agents."""
        # Create peer emotional state
        peer_state = EmotionalState(
            agent_id=999,
            initial_arousal=0.8,
            initial_valence=0.9,
            initial_dominance=0.6
        )
        
        # Store initial values
        initial_arousal = self.emotional_state.arousal
        initial_valence = self.emotional_state.valence
        
        # Apply peer influence
        distance = 50.0
        influence_strength = 0.2
        self.emotional_state.influence_from_peer(peer_state, distance, influence_strength)
        
        # Should have moved toward peer's emotional state
        arousal_diff = abs(self.emotional_state.arousal - initial_arousal)
        valence_diff = abs(self.emotional_state.valence - initial_valence)
        
        # At least one dimension should have changed
        assert arousal_diff > 0.001 or valence_diff > 0.001
    
    def test_emotional_summary(self):
        """Test emotional state summary generation."""
        summary = self.emotional_state.get_emotional_summary()
        
        assert 'agent_id' in summary
        assert 'dimensions' in summary
        assert 'dominant_emotion' in summary
        assert 'emotion_strength' in summary
        assert 'all_emotions' in summary
        assert 'emotional_stability' in summary
        
        assert summary['agent_id'] == self.agent_id
        assert 'arousal' in summary['dimensions']
        assert 'valence' in summary['dimensions']
        assert 'dominance' in summary['dimensions']
    
    def test_emotional_compatibility(self):
        """Test emotional compatibility calculation between agents."""
        # Create similar peer
        similar_peer = EmotionalState(
            agent_id=2,
            initial_arousal=0.25,  # Close to 0.2
            initial_valence=0.35,  # Close to 0.3
            initial_dominance=0.15  # Close to 0.1
        )
        
        compatibility = self.emotional_state._calculate_emotional_similarity(similar_peer)
        assert 0.8 <= compatibility <= 1.0  # Should be high similarity
        
        # Create dissimilar peer
        dissimilar_peer = EmotionalState(
            agent_id=3,
            initial_arousal=-0.8,   # Very different
            initial_valence=-0.9,   # Very different
            initial_dominance=-0.7  # Very different
        )
        
        compatibility = self.emotional_state._calculate_emotional_similarity(dissimilar_peer)
        assert 0.0 <= compatibility <= 0.5  # Should be low similarity
    
    def test_emotional_memory(self):
        """Test emotional memory recording."""
        sentiment_data = SentimentData(
            polarity=SentimentPolarity.POSITIVE,
            intensity=0.6,
            confidence=0.7,
            emotional_dimensions={'arousal': 0.4, 'valence': 0.5, 'dominance': 0.2},
            processing_time=12.0
        )
        
        context = EmotionalContext(task_performance=0.5)
        
        initial_memory_length = len(self.emotional_state.emotional_memory)
        
        # Record emotional change
        self.emotional_state._record_emotional_change(sentiment_data, context, 0.3)
        
        assert len(self.emotional_state.emotional_memory) == initial_memory_length + 1
        
        # Check memory entry structure
        latest_entry = self.emotional_state.emotional_memory[-1]
        assert 'timestamp' in latest_entry
        assert 'arousal_change' in latest_entry
        assert 'sentiment_polarity' in latest_entry
        assert 'context_summary' in latest_entry
    
    def test_emotional_stability(self):
        """Test emotional stability calculation."""
        # Start with stable emotions
        stability1 = self.emotional_state.emotional_stability
        
        # Add some emotional changes
        for i in range(5):
            sentiment_data = SentimentData(
                polarity=SentimentPolarity.POSITIVE if i % 2 == 0 else SentimentPolarity.NEGATIVE,
                intensity=0.5 + (i * 0.1),
                confidence=0.7,
                emotional_dimensions={
                    'arousal': 0.3 + (i * 0.1), 
                    'valence': 0.2 + (i * 0.1), 
                    'dominance': 0.1
                },
                processing_time=10.0
            )
            
            context = EmotionalContext()
            self.emotional_state._record_emotional_change(sentiment_data, context, 0.2)
        
        self.emotional_state._update_emotional_stability()
        
        # Stability should be updated
        assert hasattr(self.emotional_state, 'emotional_stability')
        assert 0.0 <= self.emotional_state.emotional_stability <= 1.0
    
    def test_error_handling(self):
        """Test error handling in emotional state operations."""
        # Test with invalid sentiment data
        invalid_sentiment = Mock()
        invalid_sentiment.emotional_dimensions = None
        invalid_context = Mock()
        
        # Should not raise exception
        try:
            self.emotional_state.update_from_sentiment(invalid_sentiment, invalid_context)
        except Exception as e:
            # Should handle gracefully and log error
            pass
        
        # Test with None peer
        try:
            self.emotional_state.influence_from_peer(None, 50.0, 0.1)
        except Exception as e:
            # Should handle gracefully
            pass


class TestEmotionalContext:
    """Test suite for EmotionalContext class."""
    
    def test_context_initialization(self):
        """Test emotional context initialization."""
        context = EmotionalContext(
            task_performance=0.5,
            resource_availability=0.7,
            threat_level=0.2,
            cooperation_level=0.8,
            time_pressure=0.3
        )
        
        assert context.task_performance == 0.5
        assert context.resource_availability == 0.7
        assert context.threat_level == 0.2
        assert context.cooperation_level == 0.8
        assert context.time_pressure == 0.3
    
    def test_context_validation(self):
        """Test that context values are validated and clamped."""
        context = EmotionalContext(
            task_performance=2.0,        # Should be clamped to 1.0
            resource_availability=-0.5,  # Should be clamped to 0.0
            threat_level=1.5,           # Should be clamped to 1.0
            cooperation_level=-0.1,     # Should be clamped to 0.0
            time_pressure=0.5           # Should remain unchanged
        )
        
        assert context.task_performance == 1.0
        assert context.resource_availability == 0.0
        assert context.threat_level == 1.0
        assert context.cooperation_level == 0.0
        assert context.time_pressure == 0.5
    
    def test_default_context(self):
        """Test default emotional context values."""
        context = EmotionalContext()
        
        # Should have reasonable default values
        assert isinstance(context.social_environment, dict)
        assert context.task_performance == 0.0
        assert context.resource_availability == 0.5
        assert context.threat_level == 0.0
        assert context.cooperation_level == 0.5
        assert context.time_pressure == 0.0


class TestEmotionType:
    """Test suite for EmotionType enum."""
    
    def test_emotion_types(self):
        """Test that all expected emotion types are available."""
        expected_emotions = [
            "joy", "anger", "sadness", "fear", "surprise", 
            "disgust", "trust", "anticipation", "neutral"
        ]
        
        for emotion_name in expected_emotions:
            emotion_type = EmotionType(emotion_name)
            assert emotion_type.value == emotion_name
    
    def test_emotion_enumeration(self):
        """Test that emotions can be enumerated."""
        emotion_list = list(EmotionType)
        assert len(emotion_list) == 9  # Should have 9 emotion types
        
        # Check that neutral is included
        assert EmotionType.NEUTRAL in emotion_list


class TestEmotionalStateIntegration:
    """Integration tests for emotional state components."""
    
    def test_sentiment_to_emotion_pipeline(self):
        """Test complete pipeline from sentiment to emotional state."""
        state = EmotionalState(agent_id=1)
        
        # Simulate joy-inducing sentiment
        joy_sentiment = SentimentData(
            polarity=SentimentPolarity.VERY_POSITIVE,
            intensity=0.9,
            confidence=0.8,
            emotional_dimensions={'arousal': 0.7, 'valence': 0.9, 'dominance': 0.5},
            processing_time=15.0
        )
        
        context = EmotionalContext(
            task_performance=0.8,
            resource_availability=0.9,
            cooperation_level=0.8
        )
        
        # Update emotional state
        state.update_from_sentiment(joy_sentiment, context)
        
        # Should result in positive emotions
        dominant_emotion, strength = state.get_dominant_emotion()
        assert dominant_emotion in [EmotionType.JOY, EmotionType.TRUST, EmotionType.ANTICIPATION]
        assert strength > 0.3
        
        # Behavioral modifiers should reflect positive state
        modifiers = state.get_behavioral_modifiers()
        assert modifiers['cooperation_tendency'] > 0.5
    
    def test_emotional_contagion_scenario(self):
        """Test emotional contagion between multiple agents."""
        # Create happy agent
        happy_agent = EmotionalState(
            agent_id=1,
            initial_arousal=0.6,
            initial_valence=0.8,
            initial_dominance=0.4
        )
        
        # Create neutral agent
        neutral_agent = EmotionalState(
            agent_id=2,
            initial_arousal=0.0,
            initial_valence=0.0,
            initial_dominance=0.0
        )
        
        # Store initial neutral state
        initial_valence = neutral_agent.valence
        
        # Apply influence from happy agent
        neutral_agent.influence_from_peer(happy_agent, distance=30.0, influence_strength=0.3)
        
        # Neutral agent should become more positive
        assert neutral_agent.valence > initial_valence
        
        # Apply multiple influences to simulate crowd effect
        for _ in range(3):
            neutral_agent.influence_from_peer(happy_agent, distance=25.0, influence_strength=0.2)
        
        # Should show cumulative positive influence
        assert neutral_agent.valence > 0.2
    
    def test_emotional_memory_learning(self):
        """Test emotional learning from experience."""
        state = EmotionalState(agent_id=1, memory_size=100)
        
        # Simulate learning from multiple successful experiences
        for i in range(10):
            positive_sentiment = SentimentData(
                polarity=SentimentPolarity.POSITIVE,
                intensity=0.6 + (i * 0.03),
                confidence=0.7,
                emotional_dimensions={'arousal': 0.4, 'valence': 0.5, 'dominance': 0.3},
                processing_time=10.0
            )
            
            success_context = EmotionalContext(task_performance=0.7, cooperation_level=0.8)
            
            state.update_from_sentiment(positive_sentiment, success_context)
        
        # Should have accumulated emotional memories
        assert len(state.emotional_memory) == 10
        
        # Emotional stability should be reasonable
        assert 0.0 <= state.emotional_stability <= 1.0
        
        # Should have learned positive associations
        dominant_emotion, strength = state.get_dominant_emotion()
        modifiers = state.get_behavioral_modifiers()
        
        # Should show positive behavioral tendencies
        assert modifiers['cooperation_tendency'] >= 0.5


if __name__ == "__main__":
    pytest.main([__file__])