"""
Sentiment memory management for multi-agent systems.

Provides efficient storage and retrieval of emotional experiences,
enabling long-term emotional learning and pattern recognition.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import time
import pickle
import os

from .processor import SentimentData
from .emotional_state import EmotionalContext
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmotionalMemoryEntry:
    """Single entry in emotional memory system."""
    
    timestamp: float
    sentiment_data: SentimentData
    emotional_dimensions: Dict[str, float]
    context: EmotionalContext
    action_taken: int
    reward_received: float
    peer_emotions: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate memory entry data."""
        if self.reward_received is None:
            self.reward_received = 0.0
        if not isinstance(self.action_taken, int):
            self.action_taken = 0


class SentimentMemoryBuffer:
    """
    Circular buffer for storing and analyzing emotional experiences.
    
    Provides efficient storage, pattern recognition, and emotional learning
    capabilities for sentiment-aware agents.
    """
    
    def __init__(self, capacity: int = 1000, agent_id: int = 0):
        """
        Initialize sentiment memory buffer.
        
        Args:
            capacity: Maximum number of memory entries to store
            agent_id: ID of the agent owning this memory
        """
        self.capacity = capacity
        self.agent_id = agent_id
        self.memory = deque(maxlen=capacity)
        self.emotional_patterns = {}
        self.successful_strategies = {}
        
        # Performance tracking
        self.total_entries = 0
        self.pattern_recognition_enabled = True
        self.learning_enabled = True
        
        logger.debug(f"SentimentMemoryBuffer initialized for agent {agent_id} with capacity {capacity}")
    
    def add_experience(self, sentiment_data: SentimentData, emotional_dimensions: Dict[str, float],
                      context: EmotionalContext, action_taken: int, reward_received: float,
                      peer_emotions: Optional[Dict[int, Dict[str, float]]] = None) -> None:
        """
        Add new emotional experience to memory.
        
        Args:
            sentiment_data: Sentiment analysis results
            emotional_dimensions: Current emotional state dimensions
            context: Environmental context
            action_taken: Action chosen by agent
            reward_received: Reward received for action
            peer_emotions: Emotions of nearby agents
        """
        try:
            if peer_emotions is None:
                peer_emotions = {}
            
            entry = EmotionalMemoryEntry(
                timestamp=time.time(),
                sentiment_data=sentiment_data,
                emotional_dimensions=emotional_dimensions.copy(),
                context=context,
                action_taken=action_taken,
                reward_received=reward_received,
                peer_emotions=peer_emotions.copy()
            )
            
            self.memory.append(entry)
            self.total_entries += 1
            
            # Trigger pattern analysis if enabled
            if self.pattern_recognition_enabled and len(self.memory) % 50 == 0:
                self._analyze_emotional_patterns()
            
            # Update successful strategies if learning enabled
            if self.learning_enabled and reward_received > 0.5:
                self._update_successful_strategies(entry)
                
        except Exception as e:
            logger.error(f"Adding emotional experience failed for agent {self.agent_id}: {str(e)}")
    
    def get_emotional_trends(self, window_size: int = 100) -> Dict[str, List[float]]:
        """
        Get recent emotional trends over specified window.
        
        Args:
            window_size: Number of recent entries to analyze
            
        Returns:
            Dictionary of emotional dimension trends
        """
        try:
            if not self.memory:
                return {'arousal': [0.0], 'valence': [0.0], 'dominance': [0.0]}
            
            recent_entries = list(self.memory)[-window_size:]
            
            trends = {
                'arousal': [],
                'valence': [],
                'dominance': [],
                'rewards': [],
                'timestamps': []
            }
            
            for entry in recent_entries:
                trends['arousal'].append(entry.emotional_dimensions.get('arousal', 0.0))
                trends['valence'].append(entry.emotional_dimensions.get('valence', 0.0))
                trends['dominance'].append(entry.emotional_dimensions.get('dominance', 0.0))
                trends['rewards'].append(entry.reward_received)
                trends['timestamps'].append(entry.timestamp)
            
            return trends
            
        except Exception as e:
            logger.error(f"Getting emotional trends failed for agent {self.agent_id}: {str(e)}")
            return {'arousal': [0.0], 'valence': [0.0], 'dominance': [0.0]}
    
    def find_similar_situations(self, current_context: EmotionalContext, 
                               similarity_threshold: float = 0.7) -> List[EmotionalMemoryEntry]:
        """
        Find past experiences similar to current situation.
        
        Args:
            current_context: Current environmental context
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of similar past experiences
        """
        try:
            similar_entries = []
            
            for entry in self.memory:
                similarity = self._calculate_context_similarity(current_context, entry.context)
                
                if similarity >= similarity_threshold:
                    similar_entries.append(entry)
            
            # Sort by similarity (most similar first)
            similar_entries.sort(key=lambda e: self._calculate_context_similarity(current_context, e.context), 
                               reverse=True)
            
            return similar_entries[:10]  # Return top 10 most similar
            
        except Exception as e:
            logger.error(f"Finding similar situations failed for agent {self.agent_id}: {str(e)}")
            return []
    
    def get_emotional_learning_insights(self) -> Dict[str, Any]:
        """
        Extract learning insights from emotional memory.
        
        Returns:
            Dictionary containing emotional learning insights
        """
        try:
            if len(self.memory) < 10:
                return {'insufficient_data': True}
            
            insights = {
                'total_experiences': len(self.memory),
                'average_reward': np.mean([entry.reward_received for entry in self.memory]),
                'emotional_stability': self._calculate_emotional_stability(),
                'most_rewarding_emotion': self._find_most_rewarding_emotion(),
                'successful_patterns': list(self.successful_strategies.keys()),
                'context_preferences': self._analyze_context_preferences(),
                'peer_influence_effects': self._analyze_peer_influences()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Getting emotional learning insights failed for agent {self.agent_id}: {str(e)}")
            return {'error': str(e)}
    
    def predict_optimal_emotion(self, context: EmotionalContext) -> Dict[str, float]:
        """
        Predict optimal emotional state for given context based on past experience.
        
        Args:
            context: Environmental context to predict for
            
        Returns:
            Dictionary of recommended emotional dimensions
        """
        try:
            # Find similar past situations
            similar_entries = self.find_similar_situations(context, similarity_threshold=0.5)
            
            if not similar_entries:
                # No similar situations, return neutral recommendation
                return {'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0}
            
            # Weight experiences by reward and recency
            weighted_emotions = {'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0}
            total_weight = 0.0
            current_time = time.time()
            
            for entry in similar_entries:
                # Calculate weight based on reward and recency
                reward_weight = max(0.1, entry.reward_received + 1.0)  # Shift to positive
                time_weight = max(0.1, 1.0 - (current_time - entry.timestamp) / 86400)  # Decay over 24 hours
                total_weight_entry = reward_weight * time_weight
                
                # Add weighted emotional dimensions
                weighted_emotions['arousal'] += entry.emotional_dimensions.get('arousal', 0.0) * total_weight_entry
                weighted_emotions['valence'] += entry.emotional_dimensions.get('valence', 0.0) * total_weight_entry
                weighted_emotions['dominance'] += entry.emotional_dimensions.get('dominance', 0.0) * total_weight_entry
                
                total_weight += total_weight_entry
            
            # Normalize by total weight
            if total_weight > 0:
                for dimension in weighted_emotions:
                    weighted_emotions[dimension] /= total_weight
                    weighted_emotions[dimension] = max(-1.0, min(1.0, weighted_emotions[dimension]))
            
            return weighted_emotions
            
        except Exception as e:
            logger.error(f"Predicting optimal emotion failed for agent {self.agent_id}: {str(e)}")
            return {'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0}
    
    def clear_memory(self, keep_recent: int = 0) -> None:
        """
        Clear memory buffer, optionally keeping recent entries.
        
        Args:
            keep_recent: Number of recent entries to keep
        """
        try:
            if keep_recent > 0 and keep_recent < len(self.memory):
                recent_entries = list(self.memory)[-keep_recent:]
                self.memory.clear()
                for entry in recent_entries:
                    self.memory.append(entry)
            else:
                self.memory.clear()
                
            self.emotional_patterns.clear()
            self.successful_strategies.clear()
            
            logger.info(f"Memory cleared for agent {self.agent_id}, kept {len(self.memory)} recent entries")
            
        except Exception as e:
            logger.error(f"Clearing memory failed for agent {self.agent_id}: {str(e)}")
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save memory buffer to file for persistence.
        
        Args:
            filepath: Path to save memory file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            memory_data = {
                'agent_id': self.agent_id,
                'capacity': self.capacity,
                'memory_entries': list(self.memory),
                'emotional_patterns': self.emotional_patterns,
                'successful_strategies': self.successful_strategies,
                'total_entries': self.total_entries
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(memory_data, f)
            
            logger.info(f"Memory saved to {filepath} for agent {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Saving memory to file failed for agent {self.agent_id}: {str(e)}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load memory buffer from file.
        
        Args:
            filepath: Path to memory file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Memory file {filepath} does not exist")
                return False
            
            with open(filepath, 'rb') as f:
                memory_data = pickle.load(f)
            
            self.agent_id = memory_data.get('agent_id', self.agent_id)
            self.capacity = memory_data.get('capacity', self.capacity)
            self.emotional_patterns = memory_data.get('emotional_patterns', {})
            self.successful_strategies = memory_data.get('successful_strategies', {})
            self.total_entries = memory_data.get('total_entries', 0)
            
            # Restore memory entries
            self.memory.clear()
            for entry in memory_data.get('memory_entries', []):
                self.memory.append(entry)
            
            logger.info(f"Memory loaded from {filepath} for agent {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Loading memory from file failed for agent {self.agent_id}: {str(e)}")
            return False
    
    def _analyze_emotional_patterns(self) -> None:
        """Analyze emotional patterns in memory for pattern recognition."""
        try:
            if len(self.memory) < 20:
                return
            
            # Analyze sequences of emotional states
            recent_entries = list(self.memory)[-100:]  # Last 100 entries
            
            # Look for emotional state transitions
            transitions = {}
            for i in range(len(recent_entries) - 1):
                current_emotion = self._categorize_emotion(recent_entries[i].emotional_dimensions)
                next_emotion = self._categorize_emotion(recent_entries[i + 1].emotional_dimensions)
                
                transition_key = f"{current_emotion} -> {next_emotion}"
                if transition_key not in transitions:
                    transitions[transition_key] = {'count': 0, 'avg_reward': 0.0}
                
                transitions[transition_key]['count'] += 1
                transitions[transition_key]['avg_reward'] += recent_entries[i + 1].reward_received
            
            # Calculate average rewards for transitions
            for transition in transitions:
                if transitions[transition]['count'] > 0:
                    transitions[transition]['avg_reward'] /= transitions[transition]['count']
            
            self.emotional_patterns['transitions'] = transitions
            
        except Exception as e:
            logger.warning(f"Emotional pattern analysis failed for agent {self.agent_id}: {str(e)}")
    
    def _update_successful_strategies(self, entry: EmotionalMemoryEntry) -> None:
        """Update successful emotional strategies based on high-reward experiences."""
        try:
            emotion_category = self._categorize_emotion(entry.emotional_dimensions)
            action_context = f"{emotion_category}_action_{entry.action_taken}"
            
            if action_context not in self.successful_strategies:
                self.successful_strategies[action_context] = {
                    'count': 0,
                    'total_reward': 0.0,
                    'avg_reward': 0.0,
                    'emotional_dims': {'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0}
                }
            
            strategy = self.successful_strategies[action_context]
            strategy['count'] += 1
            strategy['total_reward'] += entry.reward_received
            strategy['avg_reward'] = strategy['total_reward'] / strategy['count']
            
            # Update average emotional dimensions for this strategy
            for dim in ['arousal', 'valence', 'dominance']:
                current_avg = strategy['emotional_dims'][dim]
                new_value = entry.emotional_dimensions.get(dim, 0.0)
                strategy['emotional_dims'][dim] = (current_avg * (strategy['count'] - 1) + new_value) / strategy['count']
            
        except Exception as e:
            logger.warning(f"Updating successful strategies failed for agent {self.agent_id}: {str(e)}")
    
    def _calculate_context_similarity(self, context1: EmotionalContext, 
                                     context2: EmotionalContext) -> float:
        """Calculate similarity between two emotional contexts."""
        try:
            # Compare key context features
            similarities = []
            
            # Task performance similarity
            perf_diff = abs(context1.task_performance - context2.task_performance)
            similarities.append(1.0 - perf_diff)
            
            # Resource availability similarity
            resource_diff = abs(context1.resource_availability - context2.resource_availability)
            similarities.append(1.0 - resource_diff)
            
            # Threat level similarity
            threat_diff = abs(context1.threat_level - context2.threat_level)
            similarities.append(1.0 - threat_diff)
            
            # Cooperation level similarity
            coop_diff = abs(context1.cooperation_level - context2.cooperation_level)
            similarities.append(1.0 - coop_diff)
            
            # Time pressure similarity
            time_diff = abs(context1.time_pressure - context2.time_pressure)
            similarities.append(1.0 - time_diff)
            
            # Average similarity
            return np.mean(similarities)
            
        except Exception as e:
            logger.warning(f"Context similarity calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_emotional_stability(self) -> float:
        """Calculate emotional stability based on memory."""
        try:
            if len(self.memory) < 10:
                return 1.0
            
            # Calculate variance in emotional dimensions
            arousal_values = [entry.emotional_dimensions.get('arousal', 0.0) for entry in self.memory]
            valence_values = [entry.emotional_dimensions.get('valence', 0.0) for entry in self.memory]
            dominance_values = [entry.emotional_dimensions.get('dominance', 0.0) for entry in self.memory]
            
            arousal_var = np.var(arousal_values)
            valence_var = np.var(valence_values)
            dominance_var = np.var(dominance_values)
            
            avg_variance = (arousal_var + valence_var + dominance_var) / 3.0
            
            # Stability inversely related to variance
            stability = max(0.0, min(1.0, 1.0 - avg_variance))
            return stability
            
        except Exception as e:
            logger.warning(f"Emotional stability calculation failed: {str(e)}")
            return 0.5
    
    def _find_most_rewarding_emotion(self) -> str:
        """Find the emotional state associated with highest average rewards."""
        try:
            if not self.memory:
                return "neutral"
            
            emotion_rewards = {}
            
            for entry in self.memory:
                emotion_category = self._categorize_emotion(entry.emotional_dimensions)
                
                if emotion_category not in emotion_rewards:
                    emotion_rewards[emotion_category] = []
                
                emotion_rewards[emotion_category].append(entry.reward_received)
            
            # Calculate average rewards per emotion
            avg_rewards = {}
            for emotion, rewards in emotion_rewards.items():
                avg_rewards[emotion] = np.mean(rewards)
            
            # Find emotion with highest average reward
            best_emotion = max(avg_rewards, key=avg_rewards.get)
            return best_emotion
            
        except Exception as e:
            logger.warning(f"Finding most rewarding emotion failed: {str(e)}")
            return "neutral"
    
    def _analyze_context_preferences(self) -> Dict[str, Any]:
        """Analyze preferred contexts based on emotional memory."""
        try:
            if len(self.memory) < 10:
                return {}
            
            high_reward_entries = [entry for entry in self.memory if entry.reward_received > 0.5]
            
            if not high_reward_entries:
                return {}
            
            # Analyze context features in high-reward situations
            preferences = {
                'resource_availability': np.mean([entry.context.resource_availability for entry in high_reward_entries]),
                'cooperation_level': np.mean([entry.context.cooperation_level for entry in high_reward_entries]),
                'threat_level': np.mean([entry.context.threat_level for entry in high_reward_entries]),
                'time_pressure': np.mean([entry.context.time_pressure for entry in high_reward_entries])
            }
            
            return preferences
            
        except Exception as e:
            logger.warning(f"Context preferences analysis failed: {str(e)}")
            return {}
    
    def _analyze_peer_influences(self) -> Dict[str, Any]:
        """Analyze effects of peer emotions on agent performance."""
        try:
            if len(self.memory) < 10:
                return {}
            
            # Find entries with peer emotion data
            peer_entries = [entry for entry in self.memory if entry.peer_emotions]
            
            if not peer_entries:
                return {'no_peer_data': True}
            
            # Analyze correlation between peer emotions and rewards
            influences = {
                'positive_peer_influence': 0,
                'negative_peer_influence': 0,
                'neutral_peer_influence': 0
            }
            
            for entry in peer_entries:
                # Calculate average peer valence
                peer_valences = []
                for peer_id, peer_emotion in entry.peer_emotions.items():
                    peer_valences.append(peer_emotion.get('valence', 0.0))
                
                if peer_valences:
                    avg_peer_valence = np.mean(peer_valences)
                    
                    if avg_peer_valence > 0.3 and entry.reward_received > 0.3:
                        influences['positive_peer_influence'] += 1
                    elif avg_peer_valence < -0.3 and entry.reward_received < -0.3:
                        influences['negative_peer_influence'] += 1
                    else:
                        influences['neutral_peer_influence'] += 1
            
            return influences
            
        except Exception as e:
            logger.warning(f"Peer influence analysis failed: {str(e)}")
            return {}
    
    def _categorize_emotion(self, emotional_dimensions: Dict[str, float]) -> str:
        """Categorize emotional dimensions into discrete emotion label."""
        arousal = emotional_dimensions.get('arousal', 0.0)
        valence = emotional_dimensions.get('valence', 0.0)
        dominance = emotional_dimensions.get('dominance', 0.0)
        
        # Simple categorization based on dimensional values
        if valence > 0.3:
            if arousal > 0.3:
                return "excited" if dominance > 0.2 else "happy"
            else:
                return "content" if dominance > 0.0 else "peaceful"
        elif valence < -0.3:
            if arousal > 0.3:
                return "angry" if dominance > 0.2 else "anxious"
            else:
                return "sad" if dominance < -0.2 else "disappointed"
        else:
            if arousal > 0.5:
                return "alert"
            elif arousal < -0.3:
                return "calm"
            else:
                return "neutral"