"""
Sentiment-aware agents for emotional multi-agent reinforcement learning.

Extends base agent classes with emotional intelligence, sentiment-driven
decision making, and advanced behavioral modulation capabilities.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
from collections import deque

from .agent import BaseAgent, Agent, AgentState
from ..sentiment.processor import SentimentProcessor, SentimentData
from ..sentiment.emotional_state import EmotionalState, EmotionalContext, EmotionType
from ..sentiment.memory import SentimentMemoryBuffer
from ..sentiment.contagion import ContagionParameters
from ..utils.logging import get_logger
from ..exceptions import AgentError

logger = get_logger(__name__)


@dataclass
class SentimentAwareAgentConfig:
    """Configuration for sentiment-aware agents."""
    
    emotional_sensitivity: float = 0.7  # How much emotions affect behavior (0.0-1.0)
    sentiment_processing: str = "behavioral"  # "behavioral", "text", "hybrid"
    memory_capacity: int = 500  # Emotional memory buffer size
    contagion_susceptibility: float = 0.5  # Susceptibility to emotional contagion (0.0-1.0)
    empathy_level: float = 0.6  # Level of empathy for other agents (0.0-1.0)
    emotional_decay_rate: float = 0.95  # Rate of emotional decay (0.8-0.99)
    learning_from_emotion: bool = True  # Whether to learn from emotional experiences
    emotion_expression: bool = True  # Whether agent expresses emotions to others
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self.emotional_sensitivity = max(0.0, min(1.0, self.emotional_sensitivity))
        self.contagion_susceptibility = max(0.0, min(1.0, self.contagion_susceptibility))
        self.empathy_level = max(0.0, min(1.0, self.empathy_level))
        self.emotional_decay_rate = max(0.8, min(0.99, self.emotional_decay_rate))
        self.memory_capacity = max(10, min(2000, self.memory_capacity))


class SentimentAwareAgent(BaseAgent):
    """
    Base class for agents with emotional intelligence and sentiment analysis.
    
    Integrates sentiment processing, emotional state management, memory,
    and sentiment-driven decision making into the agent architecture.
    """
    
    def __init__(self, agent_id: int, initial_position: np.ndarray, 
                 config: Optional[SentimentAwareAgentConfig] = None, **kwargs: Any) -> None:
        """
        Initialize sentiment-aware agent.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_position: Starting position [x, y]
            config: Sentiment-aware agent configuration
            **kwargs: Additional agent parameters
        """
        super().__init__(agent_id, initial_position, **kwargs)
        
        self.sentiment_config = config or SentimentAwareAgentConfig()
        
        # Initialize sentiment processing components
        self.sentiment_processor = SentimentProcessor(
            model_type="lightweight" if self.sentiment_config.sentiment_processing != "research" else "robust"
        )
        
        self.emotional_state = EmotionalState(
            agent_id=agent_id,
            decay_rate=self.sentiment_config.emotional_decay_rate,
            memory_size=50
        )
        
        self.sentiment_memory = SentimentMemoryBuffer(
            capacity=self.sentiment_config.memory_capacity,
            agent_id=agent_id
        )
        
        # Behavioral tracking for sentiment analysis
        self.recent_actions = deque(maxlen=20)
        self.recent_rewards = deque(maxlen=20)
        self.recent_contexts = deque(maxlen=10)
        
        # Peer emotion tracking for contagion
        self.peer_emotions: Dict[int, Dict[str, float]] = {}
        self.last_emotional_update = time.time()
        
        # Performance metrics
        self.emotional_decisions = 0
        self.rational_decisions = 0
        self.empathic_actions = 0
        self.total_emotional_influence_received = 0.0
        
        logger.debug(f"SentimentAwareAgent {agent_id} initialized with config: {self.sentiment_config}")
    
    def act(self, observation: Dict[str, Any]) -> int:
        """
        Enhanced action selection with emotional intelligence.
        
        Args:
            observation: Environment observation
            
        Returns:
            Selected action with emotional modulation
        """
        try:
            # Update emotional state based on current situation
            self._process_emotional_update(observation)
            
            # Get base rational action
            rational_action = self._get_rational_action(observation)
            
            # Apply emotional modulation
            emotional_action = self._apply_emotional_modulation(rational_action, observation)
            
            # Record action for behavioral analysis
            self.recent_actions.append(emotional_action)
            
            # Update decision tracking
            if emotional_action != rational_action:
                self.emotional_decisions += 1
            else:
                self.rational_decisions += 1
            
            return emotional_action
            
        except Exception as e:
            logger.error(f"Sentiment-aware action selection failed for agent {self.agent_id}: {str(e)}")
            return super().act(observation) if hasattr(super(), 'act') else 0
    
    def record_action(self, action: int, reward: float) -> None:
        """
        Enhanced action recording with emotional learning.
        
        Args:
            action: Action taken
            reward: Reward received
        """
        try:
            # Call parent recording
            super().record_action(action, reward)
            
            # Record reward for sentiment analysis
            self.recent_rewards.append(reward)
            
            # Update emotional learning if enabled
            if self.sentiment_config.learning_from_emotion and len(self.recent_contexts) > 0:
                self._update_emotional_learning(action, reward)
            
        except Exception as e:
            logger.error(f"Recording action with emotion failed for agent {self.agent_id}: {str(e)}")
    
    def apply_peer_emotional_influence(self, peer_id: int, peer_emotion: EmotionalState, 
                                     distance: float) -> None:
        """
        Apply emotional influence from nearby peer agent.
        
        Args:
            peer_id: ID of influencing agent
            peer_emotion: Emotional state of peer agent
            distance: Distance to peer agent
        """
        try:
            if self.sentiment_config.contagion_susceptibility <= 0.0:
                return
            
            # Apply influence with susceptibility scaling
            influence_strength = self.sentiment_config.contagion_susceptibility * 0.1
            
            self.emotional_state.influence_from_peer(
                peer_emotion, distance, influence_strength
            )
            
            # Record peer emotion for memory and analysis
            self.peer_emotions[peer_id] = peer_emotion.get_emotional_summary()['dimensions']
            self.total_emotional_influence_received += influence_strength
            
        except Exception as e:
            logger.error(f"Applying peer emotional influence failed for agent {self.agent_id}: {str(e)}")
    
    def get_emotional_expression(self) -> Dict[str, Any]:
        """
        Get current emotional expression for other agents to perceive.
        
        Returns:
            Dictionary containing emotional expression data
        """
        try:
            if not self.sentiment_config.emotion_expression:
                return {'unexpressive': True}
            
            dominant_emotion, strength = self.emotional_state.get_dominant_emotion()
            
            expression = {
                'agent_id': self.agent_id,
                'dominant_emotion': dominant_emotion.value,
                'emotion_strength': strength,
                'arousal': self.emotional_state.arousal,
                'valence': self.emotional_state.valence,
                'dominance': self.emotional_state.dominance,
                'expressiveness': self.sentiment_config.emotion_expression,
                'timestamp': time.time()
            }
            
            return expression
            
        except Exception as e:
            logger.error(f"Getting emotional expression failed for agent {self.agent_id}: {str(e)}")
            return {'error': str(e)}
    
    def get_sentiment_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive sentiment and emotional analytics.
        
        Returns:
            Dictionary containing sentiment analytics
        """
        try:
            analytics = {
                'agent_id': self.agent_id,
                'emotional_state': self.emotional_state.get_emotional_summary(),
                'sentiment_processor_stats': self.sentiment_processor.get_performance_stats(),
                'memory_insights': self.sentiment_memory.get_emotional_learning_insights(),
                'decision_analytics': {
                    'emotional_decisions': self.emotional_decisions,
                    'rational_decisions': self.rational_decisions,
                    'empathic_actions': self.empathic_actions,
                    'emotion_influence_ratio': self.emotional_decisions / max(1, self.emotional_decisions + self.rational_decisions),
                    'total_influence_received': self.total_emotional_influence_received
                },
                'behavioral_patterns': {
                    'recent_actions': list(self.recent_actions),
                    'recent_rewards': list(self.recent_rewards),
                    'action_consistency': np.std(list(self.recent_actions)) if len(self.recent_actions) > 1 else 0.0,
                    'reward_trend': np.mean(list(self.recent_rewards)[-5:]) if len(self.recent_rewards) >= 5 else 0.0
                },
                'peer_interactions': {
                    'tracked_peers': len(self.peer_emotions),
                    'avg_peer_valence': np.mean([emotion.get('valence', 0.0) for emotion in self.peer_emotions.values()]) if self.peer_emotions else 0.0
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Getting sentiment analytics failed for agent {self.agent_id}: {str(e)}")
            return {'agent_id': self.agent_id, 'error': str(e)}
    
    def _process_emotional_update(self, observation: Dict[str, Any]) -> None:
        """Process emotional state update based on current observation."""
        try:
            # Create emotional context from observation
            context = self._create_emotional_context(observation)
            
            # Analyze current situation sentiment
            if self.sentiment_config.sentiment_processing == "behavioral":
                sentiment_data = self.sentiment_processor.analyze_behavioral_sentiment(
                    list(self.recent_actions), context.__dict__
                )
            elif self.sentiment_config.sentiment_processing == "text":
                # For text analysis, we'd need text input (placeholder)
                text_input = f"agent at {observation.get('position', [0, 0])} with {len(observation.get('resources', []))} resources"
                sentiment_data = self.sentiment_processor.analyze_text_sentiment(text_input)
            else:  # hybrid
                behavioral_sentiment = self.sentiment_processor.analyze_behavioral_sentiment(
                    list(self.recent_actions), context.__dict__
                )
                sentiment_data = behavioral_sentiment  # Could combine with text analysis
            
            # Update emotional state
            self.emotional_state.update_from_sentiment(sentiment_data, context)
            
            # Apply natural emotional decay
            self.emotional_state.apply_emotional_decay()
            
            # Store context for learning
            self.recent_contexts.append(context)
            
            self.last_emotional_update = time.time()
            
        except Exception as e:
            logger.error(f"Processing emotional update failed for agent {self.agent_id}: {str(e)}")
    
    def _create_emotional_context(self, observation: Dict[str, Any]) -> EmotionalContext:
        """Create emotional context from observation."""
        try:
            # Extract relevant context information
            nearby_agents = observation.get('nearby_agents', [])
            resources = observation.get('resources', [])
            position = observation.get('position', [0, 0])
            arena_bounds = observation.get('arena_bounds', {'width': 1000, 'height': 1000})
            
            # Calculate task performance based on recent rewards
            task_performance = np.mean(list(self.recent_rewards)[-5:]) if len(self.recent_rewards) >= 5 else 0.0
            task_performance = max(-1.0, min(1.0, task_performance))
            
            # Calculate resource availability
            resource_density = len(resources) / max(1, len(nearby_agents) + 1)
            resource_availability = min(1.0, resource_density / 2.0)  # Normalize
            
            # Calculate cooperation level based on agent clustering
            if nearby_agents:
                # High clustering suggests cooperation
                position_array = np.array(position)
                agent_distances = [np.linalg.norm(np.array(agent_pos) - position_array) for agent_pos in nearby_agents]
                avg_distance = np.mean(agent_distances)
                cooperation_level = max(0.0, min(1.0, 1.0 - (avg_distance / 100.0)))
            else:
                cooperation_level = 0.5  # Neutral when alone
            
            # Calculate threat level (placeholder - could be based on competitive agents nearby)
            threat_level = min(1.0, len(nearby_agents) / 10.0) if len(nearby_agents) > 5 else 0.0
            
            # Calculate time pressure (placeholder - could be based on episode progress)
            time_pressure = 0.2  # Low default time pressure
            
            # Create social environment from peer emotions
            social_environment = {}
            for peer_id, peer_emotion in self.peer_emotions.items():
                social_environment[str(peer_id)] = peer_emotion.get('valence', 0.0)
            
            context = EmotionalContext(
                social_environment=social_environment,
                task_performance=task_performance,
                resource_availability=resource_availability,
                threat_level=threat_level,
                cooperation_level=cooperation_level,
                time_pressure=time_pressure
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Creating emotional context failed for agent {self.agent_id}: {str(e)}")
            return EmotionalContext()  # Return default context
    
    def _get_rational_action(self, observation: Dict[str, Any]) -> int:
        """Get rational action without emotional modulation (placeholder for subclasses)."""
        # This is a placeholder - subclasses should implement specific rational behavior
        position = observation.get('position', [0, 0])
        resources = observation.get('resources', [])
        
        # Simple resource collection behavior
        if resources:
            nearest_resource = min(resources, key=lambda r: np.linalg.norm(np.array(r) - np.array(position)))
            diff = np.array(nearest_resource) - np.array(position)
            
            if np.linalg.norm(diff) < 10:
                return 5  # collect_resource
            elif abs(diff[0]) > abs(diff[1]):
                return 4 if diff[0] > 0 else 3  # move_right or move_left
            else:
                return 1 if diff[1] > 0 else 2  # move_up or move_down
        
        return 0  # no-op
    
    def _apply_emotional_modulation(self, rational_action: int, observation: Dict[str, Any]) -> int:
        """Apply emotional modulation to rational action."""
        try:
            # Get behavioral modifiers from emotional state
            modifiers = self.emotional_state.get_behavioral_modifiers()
            
            # Apply empathy-driven modifications
            if self.sentiment_config.empathy_level > 0.5:
                empathy_action = self._consider_empathic_action(observation, modifiers)
                if empathy_action != rational_action:
                    self.empathic_actions += 1
                    return empathy_action
            
            # Apply emotional modulation to action selection
            emotion_probability = modifiers.get('decision_confidence', 0.5) * self.sentiment_config.emotional_sensitivity
            
            if np.random.random() < emotion_probability:
                # Emotionally influenced action
                dominant_emotion, strength = self.emotional_state.get_dominant_emotion()
                
                if dominant_emotion == EmotionType.FEAR and modifiers['risk_tolerance'] < 0.3:
                    # Fear-driven behavior: avoid risky actions, seek safety
                    nearby_agents = observation.get('nearby_agents', [])
                    if nearby_agents:
                        # Move towards group for safety
                        position = np.array(observation.get('position', [0, 0]))
                        center = np.mean(nearby_agents, axis=0)
                        diff = center - position
                        
                        if abs(diff[0]) > abs(diff[1]):
                            return 4 if diff[0] > 0 else 3
                        else:
                            return 1 if diff[1] > 0 else 2
                
                elif dominant_emotion == EmotionType.ANGER and modifiers['risk_tolerance'] > 0.7:
                    # Anger-driven behavior: more aggressive resource seeking
                    resources = observation.get('resources', [])
                    if resources and rational_action != 5:  # Not already collecting
                        return rational_action  # Pursue resources more aggressively
                
                elif dominant_emotion == EmotionType.JOY and modifiers['cooperation_tendency'] > 0.7:
                    # Joy-driven behavior: more cooperative
                    nearby_agents = observation.get('nearby_agents', [])
                    if nearby_agents and len(nearby_agents) > 2:
                        # Stay with group when happy
                        return 0  # no-op to stay near others
                
                elif dominant_emotion == EmotionType.SADNESS and modifiers['exploration_tendency'] < 0.3:
                    # Sadness-driven behavior: reduced exploration
                    if rational_action in [1, 2, 3, 4]:  # Movement actions
                        if np.random.random() < 0.3:
                            return 0  # Sometimes stay still when sad
            
            return rational_action
            
        except Exception as e:
            logger.error(f"Applying emotional modulation failed for agent {self.agent_id}: {str(e)}")
            return rational_action
    
    def _consider_empathic_action(self, observation: Dict[str, Any], modifiers: Dict[str, float]) -> int:
        """Consider empathy-driven actions based on peer emotional states."""
        try:
            if not self.peer_emotions or self.sentiment_config.empathy_level < 0.3:
                return self._get_rational_action(observation)
            
            # Check if any peer is in distress (negative valence)
            distressed_peers = []
            for peer_id, peer_emotion in self.peer_emotions.items():
                if peer_emotion.get('valence', 0.0) < -0.5:
                    distressed_peers.append(peer_id)
            
            if distressed_peers and modifiers['social_attraction'] > 0.6:
                # Try to help distressed peers by moving towards them
                nearby_agents = observation.get('nearby_agents', [])
                position = np.array(observation.get('position', [0, 0]))
                
                if nearby_agents:
                    # Move towards closest agent (assuming they might be distressed)
                    closest_agent = min(nearby_agents, key=lambda a: np.linalg.norm(np.array(a) - position))
                    diff = np.array(closest_agent) - position
                    
                    if np.linalg.norm(diff) > 20:  # Not too close
                        if abs(diff[0]) > abs(diff[1]):
                            return 4 if diff[0] > 0 else 3
                        else:
                            return 1 if diff[1] > 0 else 2
            
            return self._get_rational_action(observation)
            
        except Exception as e:
            logger.error(f"Considering empathic action failed for agent {self.agent_id}: {str(e)}")
            return self._get_rational_action(observation)
    
    def _update_emotional_learning(self, action: int, reward: float) -> None:
        """Update emotional learning based on action outcome."""
        try:
            if not self.recent_contexts:
                return
            
            # Get most recent context and emotional state
            recent_context = self.recent_contexts[-1]
            emotional_dimensions = {
                'arousal': self.emotional_state.arousal,
                'valence': self.emotional_state.valence,
                'dominance': self.emotional_state.dominance
            }
            
            # Analyze recent behavioral sentiment
            sentiment_data = self.sentiment_processor.analyze_behavioral_sentiment(
                list(self.recent_actions)[-5:], recent_context.__dict__
            )
            
            # Add experience to memory
            self.sentiment_memory.add_experience(
                sentiment_data=sentiment_data,
                emotional_dimensions=emotional_dimensions,
                context=recent_context,
                action_taken=action,
                reward_received=reward,
                peer_emotions=self.peer_emotions.copy()
            )
            
            # Update emotional state's adaptive parameters based on learning
            if hasattr(self.emotional_state, 'adaptation_rate'):
                if reward > 0.5:  # Good outcome
                    self.emotional_state.adaptation_rate = min(0.2, self.emotional_state.adaptation_rate * 1.05)
                elif reward < -0.5:  # Bad outcome
                    self.emotional_state.adaptation_rate = max(0.05, self.emotional_state.adaptation_rate * 0.95)
            
        except Exception as e:
            logger.error(f"Updating emotional learning failed for agent {self.agent_id}: {str(e)}")


class EmotionalCooperativeAgent(SentimentAwareAgent):
    """Cooperative agent enhanced with emotional intelligence."""
    
    def __init__(self, agent_id: int, initial_position: np.ndarray, **kwargs: Any) -> None:
        # Configure for high cooperation and empathy
        config = SentimentAwareAgentConfig(
            emotional_sensitivity=0.8,
            empathy_level=0.9,
            contagion_susceptibility=0.7,
            emotion_expression=True
        )
        super().__init__(agent_id, initial_position, config, **kwargs)
        self.cooperation_tendency = 0.9
    
    def _get_rational_action(self, observation: Dict[str, Any]) -> int:
        """Cooperative rational action with group coordination."""
        nearby_agents = observation.get('nearby_agents', [])
        position = observation.get('position', [0, 0])
        resources = observation.get('resources', [])
        
        # Prioritize group cohesion
        if nearby_agents:
            center = np.mean(nearby_agents, axis=0)
            distance_to_center = np.linalg.norm(np.array(position) - center)
            
            if distance_to_center > 50:  # Too far from group
                diff = center - np.array(position)
                if abs(diff[0]) > abs(diff[1]):
                    return 4 if diff[0] > 0 else 3
                else:
                    return 1 if diff[1] > 0 else 2
        
        # Collect resources when available
        if resources:
            nearest_resource = min(resources, key=lambda r: np.linalg.norm(np.array(r) - np.array(position)))
            distance_to_resource = np.linalg.norm(np.array(nearest_resource) - np.array(position))
            
            if distance_to_resource < 10:
                return 5  # collect_resource
            else:
                diff = np.array(nearest_resource) - np.array(position)
                if abs(diff[0]) > abs(diff[1]):
                    return 4 if diff[0] > 0 else 3
                else:
                    return 1 if diff[1] > 0 else 2
        
        return 0  # Stay with group


class EmotionalCompetitiveAgent(SentimentAwareAgent):
    """Competitive agent enhanced with emotional intelligence."""
    
    def __init__(self, agent_id: int, initial_position: np.ndarray, **kwargs: Any) -> None:
        # Configure for competitive behavior with emotional awareness
        config = SentimentAwareAgentConfig(
            emotional_sensitivity=0.6,
            empathy_level=0.3,
            contagion_susceptibility=0.4,
            emotion_expression=True
        )
        super().__init__(agent_id, initial_position, config, **kwargs)
        self.competition_tendency = 0.8
    
    def _get_rational_action(self, observation: Dict[str, Any]) -> int:
        """Competitive rational action with resource focus."""
        resources = observation.get('resources', [])
        position = observation.get('position', [0, 0])
        nearby_agents = observation.get('nearby_agents', [])
        
        # Aggressively pursue resources
        if resources:
            nearest_resource = min(resources, key=lambda r: np.linalg.norm(np.array(r) - np.array(position)))
            distance_to_resource = np.linalg.norm(np.array(nearest_resource) - np.array(position))
            
            if distance_to_resource < 10:
                return 5  # collect_resource
            else:
                diff = np.array(nearest_resource) - np.array(position)
                if abs(diff[0]) > abs(diff[1]):
                    return 4 if diff[0] > 0 else 3
                else:
                    return 1 if diff[1] > 0 else 2
        
        # Explore for resources when none visible
        return np.random.randint(1, 5)


class EmotionalAdaptiveAgent(SentimentAwareAgent):
    """Adaptive agent that learns optimal emotional strategies."""
    
    def __init__(self, agent_id: int, initial_position: np.ndarray, **kwargs: Any) -> None:
        # Configure for high adaptability and learning
        config = SentimentAwareAgentConfig(
            emotional_sensitivity=0.75,
            empathy_level=0.6,
            contagion_susceptibility=0.6,
            learning_from_emotion=True,
            memory_capacity=1000
        )
        super().__init__(agent_id, initial_position, config, **kwargs)
        
        # Adaptive behavioral weights
        self.behavioral_strategies = {
            'cooperative': 0.25,
            'competitive': 0.25,
            'exploratory': 0.25,
            'defensive': 0.25
        }
        
        self.strategy_performance = {strategy: deque(maxlen=20) for strategy in self.behavioral_strategies}
    
    def _get_rational_action(self, observation: Dict[str, Any]) -> int:
        """Adaptive rational action based on learned strategies."""
        # Select strategy based on current weights and emotional state
        emotional_modifiers = self.emotional_state.get_behavioral_modifiers()
        
        # Adjust strategy weights based on emotional state
        adjusted_strategies = self.behavioral_strategies.copy()
        
        if emotional_modifiers['cooperation_tendency'] > 0.7:
            adjusted_strategies['cooperative'] *= 1.5
        elif emotional_modifiers['cooperation_tendency'] < 0.3:
            adjusted_strategies['competitive'] *= 1.5
        
        if emotional_modifiers['exploration_tendency'] > 0.7:
            adjusted_strategies['exploratory'] *= 1.3
        
        if emotional_modifiers['risk_tolerance'] < 0.3:
            adjusted_strategies['defensive'] *= 1.4
        
        # Normalize weights
        total_weight = sum(adjusted_strategies.values())
        for strategy in adjusted_strategies:
            adjusted_strategies[strategy] /= total_weight
        
        # Select strategy
        strategy_names = list(adjusted_strategies.keys())
        strategy_weights = list(adjusted_strategies.values())
        selected_strategy = np.random.choice(strategy_names, p=strategy_weights)
        
        # Execute strategy-specific action
        return self._execute_strategy_action(selected_strategy, observation)
    
    def _execute_strategy_action(self, strategy: str, observation: Dict[str, Any]) -> int:
        """Execute action based on selected strategy."""
        position = observation.get('position', [0, 0])
        resources = observation.get('resources', [])
        nearby_agents = observation.get('nearby_agents', [])
        
        if strategy == 'cooperative':
            # Cooperative strategy: stay with group, share resources
            if nearby_agents:
                center = np.mean(nearby_agents, axis=0)
                diff = center - np.array(position)
                if np.linalg.norm(diff) > 30:
                    if abs(diff[0]) > abs(diff[1]):
                        return 4 if diff[0] > 0 else 3
                    else:
                        return 1 if diff[1] > 0 else 2
            return 0  # Stay in place when with group
            
        elif strategy == 'competitive':
            # Competitive strategy: aggressively collect resources
            if resources:
                nearest_resource = min(resources, key=lambda r: np.linalg.norm(np.array(r) - np.array(position)))
                if np.linalg.norm(np.array(nearest_resource) - np.array(position)) < 10:
                    return 5
                else:
                    diff = np.array(nearest_resource) - np.array(position)
                    if abs(diff[0]) > abs(diff[1]):
                        return 4 if diff[0] > 0 else 3
                    else:
                        return 1 if diff[1] > 0 else 2
            return np.random.randint(1, 5)
            
        elif strategy == 'exploratory':
            # Exploratory strategy: systematic exploration
            return np.random.randint(1, 5)
            
        elif strategy == 'defensive':
            # Defensive strategy: avoid crowded areas, collect safely
            if len(nearby_agents) > 5:  # Too crowded
                # Move away from crowd
                center = np.mean(nearby_agents, axis=0)
                diff = np.array(position) - center  # Move away from center
                if np.linalg.norm(diff) < 20:
                    if abs(diff[0]) > abs(diff[1]):
                        return 4 if diff[0] > 0 else 3
                    else:
                        return 1 if diff[1] > 0 else 2
            
            # Collect resources if safe (few agents nearby)
            if resources and len(nearby_agents) < 3:
                nearest_resource = min(resources, key=lambda r: np.linalg.norm(np.array(r) - np.array(position)))
                if np.linalg.norm(np.array(nearest_resource) - np.array(position)) < 10:
                    return 5
            
            return 0  # Stay safe
        
        return 0  # Default action