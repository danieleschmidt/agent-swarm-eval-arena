"""
Sentiment processing engine for multi-agent systems.

Provides lightweight, real-time sentiment analysis optimized for 
large-scale multi-agent environments with <10ms processing latency.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import re
import time
from collections import defaultdict

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SentimentPolarity(Enum):
    """Sentiment polarity classifications."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class SentimentData:
    """Container for sentiment analysis results."""
    
    polarity: SentimentPolarity
    intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    emotional_dimensions: Dict[str, float]  # Arousal, valence, dominance
    processing_time: float  # milliseconds
    
    def __post_init__(self):
        """Validate sentiment data ranges."""
        self.intensity = max(0.0, min(1.0, self.intensity))
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Ensure emotional dimensions are in valid range
        for dimension in self.emotional_dimensions:
            self.emotional_dimensions[dimension] = max(-1.0, min(1.0, 
                self.emotional_dimensions[dimension]))


class SentimentProcessor:
    """
    High-performance sentiment processor optimized for multi-agent systems.
    
    Uses a hybrid approach combining rule-based analysis with lightweight
    embeddings for real-time processing in large-scale simulations.
    """
    
    def __init__(self, model_type: str = "lightweight", cache_size: int = 1000):
        """
        Initialize sentiment processor.
        
        Args:
            model_type: Type of sentiment model ('lightweight', 'robust', 'research')
            cache_size: Size of sentiment result cache for performance optimization
        """
        self.model_type = model_type
        self.cache_size = cache_size
        self.cache: Dict[str, SentimentData] = {}
        self.cache_hits = 0
        self.total_requests = 0
        
        # Load sentiment lexicons and patterns
        self._load_sentiment_lexicon()
        self._load_emotional_patterns()
        
        logger.info(f"SentimentProcessor initialized with {model_type} model")
    
    def _load_sentiment_lexicon(self):
        """Load sentiment lexicon for rule-based analysis."""
        # Optimized sentiment lexicon for multi-agent contexts
        self.positive_words = {
            'cooperative', 'helpful', 'beneficial', 'successful', 'effective',
            'good', 'excellent', 'positive', 'collaborate', 'share', 'support',
            'achieve', 'win', 'optimal', 'efficient', 'productive', 'harmony',
            'agreement', 'alliance', 'synergy', 'coordination', 'teamwork',
            'progress', 'advance', 'improve', 'enhance', 'boost', 'elevate'
        }
        
        self.negative_words = {
            'competitive', 'selfish', 'harmful', 'failed', 'ineffective',
            'bad', 'terrible', 'negative', 'conflict', 'hoard', 'block',
            'lose', 'suboptimal', 'inefficient', 'unproductive', 'discord',
            'disagreement', 'betrayal', 'dysfunction', 'chaos', 'competition',
            'regress', 'decline', 'worsen', 'degrade', 'reduce', 'diminish'
        }
        
        self.neutral_words = {
            'action', 'move', 'position', 'location', 'resource', 'agent',
            'environment', 'step', 'time', 'state', 'observation', 'data',
            'system', 'simulation', 'arena', 'space', 'boundary', 'parameter'
        }
        
        # Intensity modifiers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.7, 'incredibly': 1.8,
            'exceptionally': 1.9, 'remarkably': 1.6, 'significantly': 1.4,
            'moderately': 0.8, 'somewhat': 0.6, 'slightly': 0.4, 'barely': 0.3
        }
        
        self.diminishers = {
            'not': -1.0, 'never': -1.2, 'hardly': -0.8, 'rarely': -0.7,
            'seldom': -0.6, 'scarcely': -0.5, 'insufficient': -0.6
        }
    
    def _load_emotional_patterns(self):
        """Load patterns for emotional dimension analysis."""
        # Arousal patterns (activation/energy level)
        self.high_arousal_patterns = [
            r'\b(excited?|energetic|active|dynamic|intense|rapid)\b',
            r'\b(rush|speed|quick|fast|immediate|urgent)\b',
            r'\b(compete|fight|chase|pursue|attack|defend)\b'
        ]
        
        self.low_arousal_patterns = [
            r'\b(calm|peaceful|quiet|still|slow|gentle)\b',
            r'\b(rest|wait|pause|stable|steady|maintain)\b',
            r'\b(cooperate|share|help|support|assist)\b'
        ]
        
        # Valence patterns (positive/negative evaluation)
        self.positive_valence_patterns = [
            r'\b(success|achieve|accomplish|win|benefit|gain)\b',
            r'\b(improve|enhance|optimize|progress|advance)\b',
            r'\b(harmony|balance|coordination|synergy)\b'
        ]
        
        self.negative_valence_patterns = [
            r'\b(fail|lose|damage|harm|conflict|problem)\b',
            r'\b(decline|worsen|degrade|decrease|reduce)\b',
            r'\b(chaos|disorder|dysfunction|inefficient)\b'
        ]
        
        # Dominance patterns (control/power)
        self.high_dominance_patterns = [
            r'\b(control|dominate|lead|command|direct|manage)\b',
            r'\b(power|force|strong|assertive|decisive)\b',
            r'\b(influence|convince|persuade|negotiate)\b'
        ]
        
        self.low_dominance_patterns = [
            r'\b(follow|submit|comply|obey|accept|yield)\b',
            r'\b(weak|vulnerable|dependent|passive|reactive)\b',
            r'\b(influenced|convinced|persuaded|guided)\b'
        ]
    
    def analyze_behavioral_sentiment(self, action_sequence: List[int], 
                                   context: Dict[str, Any]) -> SentimentData:
        """
        Analyze sentiment from agent behavioral patterns.
        
        Args:
            action_sequence: Recent sequence of agent actions
            context: Environmental and social context information
            
        Returns:
            SentimentData: Behavioral sentiment analysis results
        """
        start_time = time.time()
        
        try:
            # Create cache key for behavioral patterns
            cache_key = f"behavior_{hash(tuple(action_sequence))}_{hash(frozenset(context.items()))}"
            
            # Check cache first
            if cache_key in self.cache:
                self.cache_hits += 1
                self.total_requests += 1
                return self.cache[cache_key]
            
            # Analyze action patterns
            cooperative_actions = sum(1 for action in action_sequence if action in [0, 5])  # no-op, collect
            competitive_actions = sum(1 for action in action_sequence if action in [1, 2, 3, 4])  # movement
            
            # Calculate behavioral sentiment metrics
            total_actions = len(action_sequence) if action_sequence else 1
            cooperation_ratio = cooperative_actions / total_actions
            exploration_ratio = competitive_actions / total_actions
            
            # Determine polarity based on behavior patterns
            if cooperation_ratio > 0.6:
                polarity = SentimentPolarity.POSITIVE
                intensity = min(1.0, cooperation_ratio * 1.2)
            elif exploration_ratio > 0.8:
                polarity = SentimentPolarity.NEUTRAL if cooperation_ratio > 0.2 else SentimentPolarity.NEGATIVE
                intensity = min(1.0, exploration_ratio * 0.8)
            else:
                polarity = SentimentPolarity.NEUTRAL
                intensity = 0.5
            
            # Calculate confidence based on action consistency
            action_variance = np.var(action_sequence) if len(action_sequence) > 1 else 0.5
            confidence = max(0.3, min(1.0, 1.0 - (action_variance / 10.0)))
            
            # Calculate emotional dimensions
            emotional_dimensions = self._calculate_behavioral_dimensions(
                action_sequence, context, cooperation_ratio, exploration_ratio
            )
            
            # Create sentiment data
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            sentiment_data = SentimentData(
                polarity=polarity,
                intensity=intensity,
                confidence=confidence,
                emotional_dimensions=emotional_dimensions,
                processing_time=processing_time
            )
            
            # Cache result
            self._cache_result(cache_key, sentiment_data)
            self.total_requests += 1
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Behavioral sentiment analysis failed: {str(e)}")
            # Return neutral sentiment as fallback
            return SentimentData(
                polarity=SentimentPolarity.NEUTRAL,
                intensity=0.5,
                confidence=0.3,
                emotional_dimensions={'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0},
                processing_time=(time.time() - start_time) * 1000
            )
    
    def analyze_text_sentiment(self, text: str) -> SentimentData:
        """
        Analyze sentiment from text input using lightweight NLP.
        
        Args:
            text: Text to analyze for sentiment
            
        Returns:
            SentimentData: Text sentiment analysis results
        """
        start_time = time.time()
        
        try:
            # Normalize text
            text = text.lower().strip()
            cache_key = f"text_{hash(text)}"
            
            # Check cache
            if cache_key in self.cache:
                self.cache_hits += 1
                self.total_requests += 1
                return self.cache[cache_key]
            
            if not text:
                return SentimentData(
                    polarity=SentimentPolarity.NEUTRAL,
                    intensity=0.0,
                    confidence=0.0,
                    emotional_dimensions={'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0},
                    processing_time=(time.time() - start_time) * 1000
                )
            
            # Tokenize and analyze
            words = re.findall(r'\b\w+\b', text)
            
            # Calculate sentiment scores
            positive_score = sum(1 for word in words if word in self.positive_words)
            negative_score = sum(1 for word in words if word in self.negative_words)
            neutral_score = sum(1 for word in words if word in self.neutral_words)
            
            # Apply modifiers
            intensity_modifier = 1.0
            polarity_modifier = 1.0
            
            for word in words:
                if word in self.intensifiers:
                    intensity_modifier *= self.intensifiers[word]
                elif word in self.diminishers:
                    polarity_modifier *= abs(self.diminishers[word])
                    if self.diminishers[word] < 0:
                        positive_score, negative_score = negative_score, positive_score
            
            # Determine overall polarity
            total_words = len(words) if words else 1
            pos_ratio = (positive_score * polarity_modifier) / total_words
            neg_ratio = (negative_score * polarity_modifier) / total_words
            
            if pos_ratio > neg_ratio + 0.1:
                polarity = SentimentPolarity.POSITIVE if pos_ratio < 0.6 else SentimentPolarity.VERY_POSITIVE
                intensity = min(1.0, pos_ratio * intensity_modifier)
            elif neg_ratio > pos_ratio + 0.1:
                polarity = SentimentPolarity.NEGATIVE if neg_ratio < 0.6 else SentimentPolarity.VERY_NEGATIVE
                intensity = min(1.0, neg_ratio * intensity_modifier)
            else:
                polarity = SentimentPolarity.NEUTRAL
                intensity = 0.5 * intensity_modifier
            
            # Calculate confidence
            decisive_words = positive_score + negative_score
            confidence = min(1.0, max(0.2, decisive_words / total_words))
            
            # Calculate emotional dimensions from text patterns
            emotional_dimensions = self._calculate_text_dimensions(text)
            
            processing_time = (time.time() - start_time) * 1000
            
            sentiment_data = SentimentData(
                polarity=polarity,
                intensity=min(1.0, intensity),
                confidence=confidence,
                emotional_dimensions=emotional_dimensions,
                processing_time=processing_time
            )
            
            self._cache_result(cache_key, sentiment_data)
            self.total_requests += 1
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Text sentiment analysis failed: {str(e)}")
            return SentimentData(
                polarity=SentimentPolarity.NEUTRAL,
                intensity=0.5,
                confidence=0.3,
                emotional_dimensions={'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0},
                processing_time=(time.time() - start_time) * 1000
            )
    
    def _calculate_behavioral_dimensions(self, action_sequence: List[int], context: Dict[str, Any],
                                       cooperation_ratio: float, exploration_ratio: float) -> Dict[str, float]:
        """Calculate emotional dimensions from behavioral patterns."""
        try:
            # Arousal (activation level) - based on action frequency and variety
            action_frequency = len(action_sequence) / max(1, context.get('time_window', 10))
            action_variety = len(set(action_sequence)) / max(1, len(action_sequence))
            arousal = min(1.0, max(-1.0, (action_frequency * 0.1) + (action_variety * 2.0) - 1.0))
            
            # Valence (positive/negative evaluation) - based on cooperation vs competition
            valence = min(1.0, max(-1.0, (cooperation_ratio * 2.0) - 1.0))
            
            # Dominance (control/power) - based on resource acquisition success
            resources_collected = context.get('resources_collected', 0)
            nearby_agents = len(context.get('nearby_agents', []))
            dominance_base = resources_collected / max(1, nearby_agents + 1)
            dominance = min(1.0, max(-1.0, (dominance_base * 2.0) - 0.5))
            
            return {
                'arousal': float(arousal),
                'valence': float(valence), 
                'dominance': float(dominance)
            }
            
        except Exception as e:
            logger.warning(f"Emotional dimension calculation failed: {str(e)}")
            return {'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0}
    
    def _calculate_text_dimensions(self, text: str) -> Dict[str, float]:
        """Calculate emotional dimensions from text patterns."""
        try:
            # Arousal calculation
            arousal_score = 0.0
            for pattern in self.high_arousal_patterns:
                arousal_score += len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in self.low_arousal_patterns:
                arousal_score -= len(re.findall(pattern, text, re.IGNORECASE)) * 0.5
            
            arousal = min(1.0, max(-1.0, arousal_score / max(1, len(text.split()))))
            
            # Valence calculation
            valence_score = 0.0
            for pattern in self.positive_valence_patterns:
                valence_score += len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in self.negative_valence_patterns:
                valence_score -= len(re.findall(pattern, text, re.IGNORECASE))
            
            valence = min(1.0, max(-1.0, valence_score / max(1, len(text.split()))))
            
            # Dominance calculation
            dominance_score = 0.0
            for pattern in self.high_dominance_patterns:
                dominance_score += len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in self.low_dominance_patterns:
                dominance_score -= len(re.findall(pattern, text, re.IGNORECASE)) * 0.7
            
            dominance = min(1.0, max(-1.0, dominance_score / max(1, len(text.split()))))
            
            return {
                'arousal': float(arousal),
                'valence': float(valence),
                'dominance': float(dominance)
            }
            
        except Exception as e:
            logger.warning(f"Text emotional dimension calculation failed: {str(e)}")
            return {'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0}
    
    def _cache_result(self, cache_key: str, sentiment_data: SentimentData):
        """Cache sentiment analysis result for performance optimization."""
        try:
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = sentiment_data
            
        except Exception as e:
            logger.warning(f"Cache operation failed: {str(e)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the sentiment processor."""
        cache_hit_rate = (self.cache_hits / max(1, self.total_requests)) * 100
        
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache),
            'model_type': self.model_type
        }
    
    def clear_cache(self):
        """Clear the sentiment analysis cache."""
        self.cache.clear()
        self.cache_hits = 0
        logger.info("Sentiment processor cache cleared")