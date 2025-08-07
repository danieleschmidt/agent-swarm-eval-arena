"""
Emotional state management for sentiment-aware agents.

Provides comprehensive emotional state tracking, decay functions,
and behavioral modulation based on emotional context.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time
from collections import deque

from .processor import SentimentData, SentimentPolarity
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EmotionType(Enum):
    """Primary emotion types for agent emotional states."""
    JOY = "joy"
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


@dataclass
class EmotionalContext:
    """Context information for emotional state updates."""
    
    social_environment: Dict[str, float] = field(default_factory=dict)  # Nearby agent emotions
    task_performance: float = 0.0  # Recent performance metric
    resource_availability: float = 0.5  # Resource scarcity/abundance
    threat_level: float = 0.0  # Environmental threat assessment
    cooperation_level: float = 0.5  # Recent cooperation success
    time_pressure: float = 0.0  # Urgency/time constraint
    
    def __post_init__(self):
        """Validate emotional context values."""
        self.task_performance = max(-1.0, min(1.0, self.task_performance))
        self.resource_availability = max(0.0, min(1.0, self.resource_availability))
        self.threat_level = max(0.0, min(1.0, self.threat_level))
        self.cooperation_level = max(0.0, min(1.0, self.cooperation_level))
        self.time_pressure = max(0.0, min(1.0, self.time_pressure))


class EmotionalState:
    """
    Comprehensive emotional state management for sentiment-aware agents.
    
    Maintains emotional dimensions (arousal, valence, dominance), primary emotions,
    emotional memory, and provides behavioral modulation based on emotional context.
    """
    
    def __init__(self, agent_id: int, initial_arousal: float = 0.0, 
                 initial_valence: float = 0.0, initial_dominance: float = 0.0,
                 decay_rate: float = 0.95, memory_size: int = 50):
        """
        Initialize emotional state.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_arousal: Starting arousal level (-1.0 to 1.0)
            initial_valence: Starting valence level (-1.0 to 1.0)
            initial_dominance: Starting dominance level (-1.0 to 1.0)
            decay_rate: Emotional decay rate per step (0.0 to 1.0)
            memory_size: Size of emotional memory buffer
        """
        self.agent_id = agent_id
        self.decay_rate = max(0.8, min(0.99, decay_rate))
        self.memory_size = memory_size
        
        # Core emotional dimensions
        self.arousal = max(-1.0, min(1.0, initial_arousal))
        self.valence = max(-1.0, min(1.0, initial_valence))
        self.dominance = max(-1.0, min(1.0, initial_dominance))
        
        # Primary emotion strengths (0.0 to 1.0)
        self.emotions = {emotion: 0.0 for emotion in EmotionType}
        self.emotions[EmotionType.NEUTRAL] = 1.0
        
        # Emotional memory and history
        self.emotional_memory = deque(maxlen=memory_size)
        self.recent_influences = deque(maxlen=10)
        
        # Behavioral modulation parameters
        self.emotional_sensitivity = 0.7  # How much emotions affect behavior
        self.adaptation_rate = 0.1  # How quickly emotions change
        self.baseline_emotion = EmotionType.NEUTRAL
        
        # Performance tracking
        self.emotional_stability = 1.0  # Measure of emotional consistency
        self.last_update_time = time.time()
        
        logger.debug(f"EmotionalState initialized for agent {agent_id}")
    
    def update_from_sentiment(self, sentiment_data: SentimentData, 
                            context: EmotionalContext) -> None:
        """
        Update emotional state based on sentiment analysis results.
        
        Args:
            sentiment_data: Results from sentiment analysis
            context: Environmental and social context
        """
        try:
            # Calculate emotional influence strength
            influence_strength = sentiment_data.intensity * sentiment_data.confidence
            
            # Update emotional dimensions with adaptive learning
            arousal_change = (sentiment_data.emotional_dimensions.get('arousal', 0.0) - self.arousal) * self.adaptation_rate
            valence_change = (sentiment_data.emotional_dimensions.get('valence', 0.0) - self.valence) * self.adaptation_rate
            dominance_change = (sentiment_data.emotional_dimensions.get('dominance', 0.0) - self.dominance) * self.adaptation_rate
            
            # Apply contextual modulation
            arousal_change *= self._calculate_arousal_modifier(context)
            valence_change *= self._calculate_valence_modifier(context)
            dominance_change *= self._calculate_dominance_modifier(context)
            
            # Update dimensions
            self.arousal = max(-1.0, min(1.0, self.arousal + arousal_change))
            self.valence = max(-1.0, min(1.0, self.valence + valence_change))
            self.dominance = max(-1.0, min(1.0, self.dominance + dominance_change))
            
            # Update primary emotions based on dimensional model
            self._update_primary_emotions()
            
            # Record emotional change in memory
            self._record_emotional_change(sentiment_data, context, influence_strength)
            
            # Update emotional stability metric
            self._update_emotional_stability()
            
            self.last_update_time = time.time()
            
        except Exception as e:
            logger.error(f"Emotional state update failed for agent {self.agent_id}: {str(e)}")
    
    def apply_emotional_decay(self) -> None:
        """Apply natural emotional decay towards baseline state."""
        try:
            # Decay emotional dimensions towards neutral
            self.arousal *= self.decay_rate
            self.valence *= self.decay_rate
            self.dominance *= self.decay_rate
            
            # Decay primary emotions towards baseline
            baseline_strength = 0.2 if self.baseline_emotion != EmotionType.NEUTRAL else 0.5
            
            for emotion_type in self.emotions:
                if emotion_type == self.baseline_emotion:
                    # Move towards baseline emotion
                    target = baseline_strength
                else:
                    # Decay other emotions
                    target = 0.0
                
                current = self.emotions[emotion_type]
                self.emotions[emotion_type] = current * self.decay_rate + target * (1 - self.decay_rate)
            
            # Ensure emotions sum to reasonable total
            self._normalize_emotions()
            
        except Exception as e:
            logger.error(f"Emotional decay failed for agent {self.agent_id}: {str(e)}")
    
    def get_dominant_emotion(self) -> Tuple[EmotionType, float]:
        """
        Get the currently dominant emotion.
        
        Returns:
            Tuple of (emotion_type, strength)
        """
        try:
            dominant_emotion = max(self.emotions.items(), key=lambda x: x[1])
            return dominant_emotion[0], dominant_emotion[1]
        except Exception as e:
            logger.error(f"Getting dominant emotion failed for agent {self.agent_id}: {str(e)}")
            return EmotionType.NEUTRAL, 0.5
    
    def get_behavioral_modifiers(self) -> Dict[str, float]:
        """
        Get behavioral modification factors based on current emotional state.
        
        Returns:
            Dictionary of behavioral modifiers for different action types
        """
        try:
            dominant_emotion, strength = self.get_dominant_emotion()
            
            # Base modifiers
            modifiers = {
                'exploration_tendency': 0.5,
                'cooperation_tendency': 0.5,
                'risk_tolerance': 0.5,
                'action_speed': 1.0,
                'decision_confidence': 0.5,
                'social_attraction': 0.5
            }
            
            # Emotional modulation based on arousal-valence model
            arousal_factor = abs(self.arousal)
            valence_factor = self.valence
            dominance_factor = self.dominance
            
            # High arousal increases action speed and exploration
            modifiers['action_speed'] *= (1.0 + arousal_factor * 0.5)
            modifiers['exploration_tendency'] += arousal_factor * 0.3
            
            # Positive valence increases cooperation and social attraction
            if valence_factor > 0:
                modifiers['cooperation_tendency'] += valence_factor * 0.4
                modifiers['social_attraction'] += valence_factor * 0.3
                modifiers['risk_tolerance'] += valence_factor * 0.2
            else:
                # Negative valence increases caution and reduces cooperation
                modifiers['cooperation_tendency'] += valence_factor * 0.3  # Reduces since valence is negative
                modifiers['risk_tolerance'] += valence_factor * 0.4
                modifiers['exploration_tendency'] += abs(valence_factor) * 0.2
            
            # Dominance affects decision confidence and social behavior
            modifiers['decision_confidence'] += dominance_factor * 0.3
            if dominance_factor > 0:
                modifiers['social_attraction'] -= dominance_factor * 0.2  # Less likely to follow others
            else:
                modifiers['social_attraction'] += abs(dominance_factor) * 0.3  # More likely to follow
            
            # Emotion-specific modifiers
            if dominant_emotion == EmotionType.JOY:
                modifiers['cooperation_tendency'] += strength * 0.3
                modifiers['exploration_tendency'] += strength * 0.2
            elif dominant_emotion == EmotionType.FEAR:
                modifiers['risk_tolerance'] -= strength * 0.4
                modifiers['social_attraction'] += strength * 0.3
                modifiers['exploration_tendency'] -= strength * 0.2
            elif dominant_emotion == EmotionType.ANGER:
                modifiers['cooperation_tendency'] -= strength * 0.4
                modifiers['risk_tolerance'] += strength * 0.3
                modifiers['action_speed'] *= (1.0 + strength * 0.3)
            elif dominant_emotion == EmotionType.TRUST:
                modifiers['cooperation_tendency'] += strength * 0.4
                modifiers['social_attraction'] += strength * 0.2
            elif dominant_emotion == EmotionType.SADNESS:
                modifiers['exploration_tendency'] -= strength * 0.3
                modifiers['action_speed'] *= (1.0 - strength * 0.2)
                modifiers['cooperation_tendency'] -= strength * 0.2
            
            # Ensure modifiers stay in reasonable bounds
            for key in modifiers:
                if key == 'action_speed':
                    modifiers[key] = max(0.1, min(3.0, modifiers[key]))
                else:
                    modifiers[key] = max(0.0, min(1.0, modifiers[key]))
            
            # Apply emotional sensitivity scaling
            for key in modifiers:
                if key != 'action_speed':
                    base_value = 0.5
                    modifiers[key] = base_value + (modifiers[key] - base_value) * self.emotional_sensitivity
            
            return modifiers
            
        except Exception as e:
            logger.error(f"Getting behavioral modifiers failed for agent {self.agent_id}: {str(e)}")
            return {
                'exploration_tendency': 0.5,
                'cooperation_tendency': 0.5,
                'risk_tolerance': 0.5,
                'action_speed': 1.0,
                'decision_confidence': 0.5,
                'social_attraction': 0.5
            }
    
    def influence_from_peer(self, peer_emotion: 'EmotionalState', 
                          distance: float, influence_strength: float = 0.1) -> None:
        """
        Apply emotional influence from nearby peer agent.
        
        Args:
            peer_emotion: Emotional state of nearby agent
            distance: Distance to peer agent (affects influence strength)
            influence_strength: Base strength of emotional influence
        """
        try:
            # Calculate distance-based influence decay
            max_influence_distance = 100.0
            distance_factor = max(0.0, 1.0 - (distance / max_influence_distance))
            
            # Calculate emotional compatibility for influence
            emotional_similarity = self._calculate_emotional_similarity(peer_emotion)
            
            # Total influence strength
            total_influence = influence_strength * distance_factor * emotional_similarity
            
            if total_influence > 0.01:  # Only apply significant influences
                # Influence emotional dimensions
                arousal_influence = (peer_emotion.arousal - self.arousal) * total_influence * 0.1
                valence_influence = (peer_emotion.valence - self.valence) * total_influence * 0.1
                dominance_influence = (peer_emotion.dominance - self.dominance) * total_influence * 0.05
                
                self.arousal += arousal_influence
                self.valence += valence_influence
                self.dominance += dominance_influence
                
                # Bound check
                self.arousal = max(-1.0, min(1.0, self.arousal))
                self.valence = max(-1.0, min(1.0, self.valence))
                self.dominance = max(-1.0, min(1.0, self.dominance))
                
                # Record influence in memory
                self.recent_influences.append({
                    'peer_id': peer_emotion.agent_id,
                    'influence_strength': total_influence,
                    'distance': distance,
                    'timestamp': time.time()
                })
                
                # Update primary emotions
                self._update_primary_emotions()
            
        except Exception as e:
            logger.error(f"Peer emotional influence failed for agent {self.agent_id}: {str(e)}")
    
    def get_emotional_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of current emotional state.
        
        Returns:
            Dictionary containing emotional state information
        """
        try:
            dominant_emotion, emotion_strength = self.get_dominant_emotion()
            
            return {
                'agent_id': self.agent_id,
                'dimensions': {
                    'arousal': float(self.arousal),
                    'valence': float(self.valence),
                    'dominance': float(self.dominance)
                },
                'dominant_emotion': dominant_emotion.value,
                'emotion_strength': float(emotion_strength),
                'all_emotions': {emotion.value: float(strength) for emotion, strength in self.emotions.items()},
                'emotional_stability': float(self.emotional_stability),
                'recent_influences': len(self.recent_influences),
                'memory_entries': len(self.emotional_memory),
                'last_update': self.last_update_time
            }
            
        except Exception as e:
            logger.error(f"Getting emotional summary failed for agent {self.agent_id}: {str(e)}")
            return {
                'agent_id': self.agent_id,
                'error': str(e)
            }
    
    def _calculate_arousal_modifier(self, context: EmotionalContext) -> float:
        """Calculate contextual modifier for arousal changes."""
        modifier = 1.0
        
        # High threat increases arousal sensitivity
        modifier += context.threat_level * 0.5
        
        # Time pressure increases arousal sensitivity
        modifier += context.time_pressure * 0.3
        
        # Social environment affects arousal
        if context.social_environment:
            avg_peer_arousal = np.mean(list(context.social_environment.values()))
            modifier += abs(avg_peer_arousal) * 0.2
        
        return max(0.5, min(2.0, modifier))
    
    def _calculate_valence_modifier(self, context: EmotionalContext) -> float:
        """Calculate contextual modifier for valence changes."""
        modifier = 1.0
        
        # Task performance strongly affects valence sensitivity
        modifier += abs(context.task_performance) * 0.4
        
        # Resource availability affects valence
        if context.resource_availability < 0.3:  # Scarcity
            modifier += 0.3
        elif context.resource_availability > 0.7:  # Abundance
            modifier += 0.2
        
        # Cooperation level affects valence sensitivity
        modifier += context.cooperation_level * 0.2
        
        return max(0.5, min(2.0, modifier))
    
    def _calculate_dominance_modifier(self, context: EmotionalContext) -> float:
        """Calculate contextual modifier for dominance changes."""
        modifier = 1.0
        
        # Task performance affects dominance sensitivity
        modifier += context.task_performance * 0.3
        
        # Threat level affects dominance
        modifier += context.threat_level * 0.2
        
        # Competition vs cooperation context
        if context.cooperation_level < 0.4:  # Competitive environment
            modifier += 0.3
        
        return max(0.5, min(2.0, modifier))
    
    def _update_primary_emotions(self) -> None:
        """Update primary emotion strengths based on dimensional values."""
        try:
            # Clear current emotions
            for emotion in self.emotions:
                self.emotions[emotion] = 0.0
            
            # Map dimensions to emotions using circumplex model
            arousal = self.arousal
            valence = self.valence
            dominance = self.dominance
            
            # High arousal, high valence emotions
            if arousal > 0.3 and valence > 0.3:
                if dominance > 0.2:
                    self.emotions[EmotionType.JOY] = min(arousal, valence) * (1.0 + dominance * 0.5)
                else:
                    self.emotions[EmotionType.SURPRISE] = min(arousal, valence) * (1.0 - dominance * 0.3)
            
            # High arousal, low valence emotions  
            elif arousal > 0.3 and valence < -0.3:
                if dominance > 0.2:
                    self.emotions[EmotionType.ANGER] = arousal * abs(valence) * (1.0 + dominance * 0.5)
                else:
                    self.emotions[EmotionType.FEAR] = arousal * abs(valence) * (1.0 - dominance * 0.5)
            
            # Low arousal, high valence emotions
            elif arousal < -0.1 and valence > 0.3:
                self.emotions[EmotionType.TRUST] = abs(arousal) * valence * (1.0 + dominance * 0.3)
            
            # Low arousal, low valence emotions
            elif arousal < -0.1 and valence < -0.3:
                self.emotions[EmotionType.SADNESS] = abs(arousal) * abs(valence) * (1.0 - dominance * 0.3)
            
            # Moderate emotions
            elif abs(arousal) <= 0.3 and abs(valence) <= 0.3:
                if dominance > 0.4:
                    self.emotions[EmotionType.ANTICIPATION] = (1.0 - abs(arousal)) * (1.0 - abs(valence)) * dominance
                else:
                    self.emotions[EmotionType.NEUTRAL] = (1.0 - abs(arousal)) * (1.0 - abs(valence)) * (1.0 - abs(dominance))
            
            # Disgust for specific combinations
            if valence < -0.5 and dominance < -0.3:
                self.emotions[EmotionType.DISGUST] = abs(valence) * abs(dominance) * 0.7
            
            # Ensure at least some emotion is present
            total_emotion = sum(self.emotions.values())
            if total_emotion < 0.1:
                self.emotions[EmotionType.NEUTRAL] = 0.5
            
            # Normalize emotions to prevent over-accumulation
            self._normalize_emotions()
            
        except Exception as e:
            logger.error(f"Primary emotion update failed for agent {self.agent_id}: {str(e)}")
            self.emotions[EmotionType.NEUTRAL] = 1.0
    
    def _normalize_emotions(self) -> None:
        """Normalize emotion strengths to reasonable values."""
        try:
            total = sum(self.emotions.values())
            if total > 1.5:  # Prevent over-accumulation
                for emotion in self.emotions:
                    self.emotions[emotion] = self.emotions[emotion] / (total * 0.8)
            elif total < 0.1:  # Ensure minimum emotional presence
                self.emotions[EmotionType.NEUTRAL] = 0.5
        except Exception as e:
            logger.error(f"Emotion normalization failed for agent {self.agent_id}: {str(e)}")
    
    def _record_emotional_change(self, sentiment_data: SentimentData, 
                                context: EmotionalContext, influence_strength: float) -> None:
        """Record emotional change in memory for analysis."""
        try:
            memory_entry = {
                'timestamp': time.time(),
                'arousal_change': sentiment_data.emotional_dimensions.get('arousal', 0.0),
                'valence_change': sentiment_data.emotional_dimensions.get('valence', 0.0),
                'dominance_change': sentiment_data.emotional_dimensions.get('dominance', 0.0),
                'sentiment_polarity': sentiment_data.polarity.value,
                'sentiment_intensity': sentiment_data.intensity,
                'influence_strength': influence_strength,
                'context_summary': {
                    'task_performance': context.task_performance,
                    'resource_availability': context.resource_availability,
                    'threat_level': context.threat_level
                }
            }
            
            self.emotional_memory.append(memory_entry)
            
        except Exception as e:
            logger.warning(f"Emotional memory recording failed for agent {self.agent_id}: {str(e)}")
    
    def _update_emotional_stability(self) -> None:
        """Update emotional stability metric based on recent changes."""
        try:
            if len(self.emotional_memory) >= 5:
                # Calculate variance in recent emotional changes
                recent_changes = list(self.emotional_memory)[-5:]
                
                arousal_changes = [entry['arousal_change'] for entry in recent_changes]
                valence_changes = [entry['valence_change'] for entry in recent_changes]
                
                arousal_variance = np.var(arousal_changes)
                valence_variance = np.var(valence_changes)
                
                # Stability inversely related to variance
                avg_variance = (arousal_variance + valence_variance) / 2.0
                self.emotional_stability = max(0.1, min(1.0, 1.0 - avg_variance))
            
        except Exception as e:
            logger.warning(f"Emotional stability update failed for agent {self.agent_id}: {str(e)}")
    
    def _calculate_emotional_similarity(self, peer_emotion: 'EmotionalState') -> float:
        """Calculate similarity between this and peer emotional state."""
        try:
            # Calculate dimensional similarity
            arousal_diff = abs(self.arousal - peer_emotion.arousal)
            valence_diff = abs(self.valence - peer_emotion.valence)
            dominance_diff = abs(self.dominance - peer_emotion.dominance)
            
            # Average difference (0 = identical, 2 = maximally different)
            avg_diff = (arousal_diff + valence_diff + dominance_diff) / 3.0
            
            # Convert to similarity (1 = identical, 0 = maximally different)
            similarity = max(0.0, 1.0 - (avg_diff / 2.0))
            
            return similarity
            
        except Exception as e:
            logger.warning(f"Emotional similarity calculation failed for agent {self.agent_id}: {str(e)}")
            return 0.5  # Neutral similarity