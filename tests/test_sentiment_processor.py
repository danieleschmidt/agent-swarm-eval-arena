"""
Unit tests for sentiment processing components.

Tests the core sentiment analysis functionality for behavioral and text-based
sentiment processing in multi-agent environments.
"""

import pytest
import numpy as np
import time
from typing import Dict, List

from swarm_arena.sentiment.processor import (
    SentimentProcessor, SentimentData, SentimentPolarity
)
from swarm_arena.sentiment.emotional_state import EmotionalContext


class TestSentimentProcessor:
    """Test suite for SentimentProcessor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = SentimentProcessor(model_type="lightweight", cache_size=100)
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.model_type == "lightweight"
        assert self.processor.cache_size == 100
        assert len(self.processor.cache) == 0
        assert self.processor.total_requests == 0
        assert self.processor.cache_hits == 0
    
    def test_behavioral_sentiment_analysis(self):
        """Test behavioral sentiment analysis."""
        # Test cooperative behavior
        cooperative_actions = [0, 5, 0, 5, 0]  # no-op and collect actions
        context = {
            'time_window': 5,
            'resources_collected': 3,
            'nearby_agents': [[100, 100], [120, 120]]
        }
        
        result = self.processor.analyze_behavioral_sentiment(cooperative_actions, context)
        
        assert isinstance(result, SentimentData)
        assert result.polarity in [SentimentPolarity.POSITIVE, SentimentPolarity.NEUTRAL]
        assert 0.0 <= result.intensity <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time > 0
        assert 'arousal' in result.emotional_dimensions
        assert 'valence' in result.emotional_dimensions
        assert 'dominance' in result.emotional_dimensions
    
    def test_competitive_behavior_sentiment(self):
        """Test sentiment analysis for competitive behavior."""
        competitive_actions = [1, 2, 3, 4, 1, 2, 3, 4]  # Movement actions
        context = {
            'time_window': 8,
            'resources_collected': 5,
            'nearby_agents': [[50, 50]]
        }
        
        result = self.processor.analyze_behavioral_sentiment(competitive_actions, context)
        
        assert isinstance(result, SentimentData)
        assert result.processing_time > 0
        # Should have some emotional dimensions
        assert all(dim in result.emotional_dimensions for dim in ['arousal', 'valence', 'dominance'])
    
    def test_text_sentiment_analysis(self):
        """Test text-based sentiment analysis."""
        # Positive text
        positive_text = "cooperative helpful successful excellent progress"
        result = self.processor.analyze_text_sentiment(positive_text)
        
        assert isinstance(result, SentimentData)
        assert result.polarity in [SentimentPolarity.POSITIVE, SentimentPolarity.VERY_POSITIVE]
        assert result.intensity > 0.0
        assert result.confidence > 0.0
        
        # Negative text
        negative_text = "failed terrible conflict harmful inefficient"
        result = self.processor.analyze_text_sentiment(negative_text)
        
        assert result.polarity in [SentimentPolarity.NEGATIVE, SentimentPolarity.VERY_NEGATIVE]
        assert result.intensity > 0.0
        
        # Neutral text
        neutral_text = "agent position location system data"
        result = self.processor.analyze_text_sentiment(neutral_text)
        
        assert result.polarity == SentimentPolarity.NEUTRAL
    
    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        # Empty action sequence
        result = self.processor.analyze_behavioral_sentiment([], {})
        assert isinstance(result, SentimentData)
        assert result.polarity == SentimentPolarity.NEUTRAL
        
        # Empty text
        result = self.processor.analyze_text_sentiment("")
        assert result.polarity == SentimentPolarity.NEUTRAL
        assert result.intensity == 0.0
        assert result.confidence == 0.0
    
    def test_cache_functionality(self):
        """Test sentiment analysis caching."""
        actions = [1, 2, 3, 4]
        context = {'time_window': 4}
        
        # First request
        result1 = self.processor.analyze_behavioral_sentiment(actions, context)
        assert self.processor.total_requests == 1
        assert self.processor.cache_hits == 0
        
        # Second identical request (should hit cache)
        result2 = self.processor.analyze_behavioral_sentiment(actions, context)
        assert self.processor.total_requests == 2
        assert self.processor.cache_hits == 1
        
        # Results should be identical
        assert result1.polarity == result2.polarity
        assert result1.intensity == result2.intensity
    
    def test_performance_stats(self):
        """Test performance statistics tracking."""
        # Generate some requests
        for i in range(5):
            self.processor.analyze_text_sentiment(f"test text {i}")
        
        stats = self.processor.get_performance_stats()
        
        assert stats['total_requests'] == 5
        assert stats['cache_hits'] == 0  # No cache hits for different texts
        assert stats['cache_hit_rate'] == 0.0
        assert stats['model_type'] == "lightweight"
        assert stats['cache_size'] >= 0
    
    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Add some entries to cache
        self.processor.analyze_text_sentiment("test1")
        self.processor.analyze_text_sentiment("test2")
        
        assert len(self.processor.cache) > 0
        
        # Clear cache
        self.processor.clear_cache()
        
        assert len(self.processor.cache) == 0
        assert self.processor.cache_hits == 0
    
    def test_emotional_dimensions_validity(self):
        """Test that emotional dimensions are within valid ranges."""
        actions = [1, 2, 3, 4, 5]
        context = {'resources_collected': 2, 'nearby_agents': []}
        
        result = self.processor.analyze_behavioral_sentiment(actions, context)
        
        for dim_name, dim_value in result.emotional_dimensions.items():
            assert -1.0 <= dim_value <= 1.0, f"Dimension {dim_name} out of range: {dim_value}"
    
    def test_data_validation(self):
        """Test SentimentData validation."""
        # Test valid data
        data = SentimentData(
            polarity=SentimentPolarity.POSITIVE,
            intensity=0.8,
            confidence=0.9,
            emotional_dimensions={'arousal': 0.5, 'valence': 0.7, 'dominance': 0.3},
            processing_time=10.0
        )
        
        assert data.intensity == 0.8
        assert data.confidence == 0.9
        assert -1.0 <= data.emotional_dimensions['arousal'] <= 1.0
        
        # Test boundary validation (should clamp values)
        data = SentimentData(
            polarity=SentimentPolarity.POSITIVE,
            intensity=1.5,  # Should be clamped to 1.0
            confidence=-0.1,  # Should be clamped to 0.0
            emotional_dimensions={'arousal': 2.0},  # Should be clamped
            processing_time=5.0
        )
        
        assert data.intensity == 1.0
        assert data.confidence == 0.0
        assert data.emotional_dimensions['arousal'] == 1.0
    
    def test_error_handling(self):
        """Test error handling in sentiment processing."""
        # Test with invalid action sequence (should handle gracefully)
        result = self.processor.analyze_behavioral_sentiment([999, 'invalid', None], {})
        assert isinstance(result, SentimentData)
        
        # Test with malformed context
        result = self.processor.analyze_behavioral_sentiment([1, 2], None)
        assert isinstance(result, SentimentData)
    
    def test_intensifier_processing(self):
        """Test processing of intensity modifiers in text."""
        # Text with intensifiers
        intense_text = "very excellent extremely successful highly effective"
        result = self.processor.analyze_text_sentiment(intense_text)
        
        # Should have higher intensity due to modifiers
        assert result.intensity > 0.5
        
        # Text with diminishers
        weak_text = "slightly good somewhat effective barely adequate"
        result = self.processor.analyze_text_sentiment(weak_text)
        
        # Should have lower intensity
        assert result.intensity < 0.8
    
    def test_negation_handling(self):
        """Test handling of negation in text sentiment."""
        positive_text = "successful effective excellent"
        negated_text = "not successful not effective not excellent"
        
        pos_result = self.processor.analyze_text_sentiment(positive_text)
        neg_result = self.processor.analyze_text_sentiment(negated_text)
        
        # Negated text should have opposite polarity
        assert pos_result.polarity.value > 0
        assert neg_result.polarity.value < 0


class TestSentimentData:
    """Test suite for SentimentData class."""
    
    def test_sentiment_data_creation(self):
        """Test SentimentData object creation and validation."""
        data = SentimentData(
            polarity=SentimentPolarity.POSITIVE,
            intensity=0.7,
            confidence=0.8,
            emotional_dimensions={'arousal': 0.5, 'valence': 0.6, 'dominance': 0.4},
            processing_time=15.5
        )
        
        assert data.polarity == SentimentPolarity.POSITIVE
        assert data.intensity == 0.7
        assert data.confidence == 0.8
        assert data.processing_time == 15.5
        assert len(data.emotional_dimensions) == 3
    
    def test_emotional_dimensions_clamping(self):
        """Test that emotional dimensions are properly clamped to [-1, 1]."""
        data = SentimentData(
            polarity=SentimentPolarity.NEUTRAL,
            intensity=0.5,
            confidence=0.5,
            emotional_dimensions={
                'arousal': 2.0,      # Should be clamped to 1.0
                'valence': -1.5,     # Should be clamped to -1.0
                'dominance': 0.5     # Should remain unchanged
            },
            processing_time=10.0
        )
        
        assert data.emotional_dimensions['arousal'] == 1.0
        assert data.emotional_dimensions['valence'] == -1.0
        assert data.emotional_dimensions['dominance'] == 0.5


@pytest.fixture
def sample_emotional_context():
    """Fixture providing a sample emotional context."""
    return EmotionalContext(
        social_environment={'peer1': 0.5, 'peer2': -0.2},
        task_performance=0.3,
        resource_availability=0.7,
        threat_level=0.1,
        cooperation_level=0.8,
        time_pressure=0.2
    )


def test_emotional_context_validation(sample_emotional_context):
    """Test emotional context validation."""
    context = sample_emotional_context
    
    # All values should be within valid ranges
    assert -1.0 <= context.task_performance <= 1.0
    assert 0.0 <= context.resource_availability <= 1.0
    assert 0.0 <= context.threat_level <= 1.0
    assert 0.0 <= context.cooperation_level <= 1.0
    assert 0.0 <= context.time_pressure <= 1.0


def test_emotional_context_clamping():
    """Test that emotional context values are clamped to valid ranges."""
    context = EmotionalContext(
        task_performance=2.0,        # Should be clamped to 1.0
        resource_availability=-0.5,  # Should be clamped to 0.0
        threat_level=1.5,            # Should be clamped to 1.0
        cooperation_level=0.5,       # Should remain unchanged
        time_pressure=-0.1           # Should be clamped to 0.0
    )
    
    assert context.task_performance == 1.0
    assert context.resource_availability == 0.0
    assert context.threat_level == 1.0
    assert context.cooperation_level == 0.5
    assert context.time_pressure == 0.0


class TestSentimentPolarity:
    """Test suite for SentimentPolarity enum."""
    
    def test_polarity_values(self):
        """Test sentiment polarity enum values."""
        assert SentimentPolarity.VERY_NEGATIVE.value == -2
        assert SentimentPolarity.NEGATIVE.value == -1
        assert SentimentPolarity.NEUTRAL.value == 0
        assert SentimentPolarity.POSITIVE.value == 1
        assert SentimentPolarity.VERY_POSITIVE.value == 2
    
    def test_polarity_ordering(self):
        """Test that sentiment polarities can be ordered."""
        polarities = [
            SentimentPolarity.VERY_NEGATIVE,
            SentimentPolarity.NEGATIVE,
            SentimentPolarity.NEUTRAL,
            SentimentPolarity.POSITIVE,
            SentimentPolarity.VERY_POSITIVE
        ]
        
        values = [p.value for p in polarities]
        assert values == sorted(values)  # Should be in ascending order


if __name__ == "__main__":
    pytest.main([__file__])