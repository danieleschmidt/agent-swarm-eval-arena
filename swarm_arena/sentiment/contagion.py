"""
Sentiment contagion algorithms for multi-agent systems.

Implements emotional spreading, influence propagation, and collective
sentiment dynamics in large-scale agent populations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
from collections import defaultdict

from .emotional_state import EmotionalState, EmotionType
from ..utils.logging import get_logger
from ..utils.spatial import calculate_distance

logger = get_logger(__name__)


@dataclass
class ContagionParameters:
    """Parameters for sentiment contagion algorithms."""
    
    base_influence_strength: float = 0.1  # Base emotional influence strength
    max_influence_distance: float = 100.0  # Maximum distance for emotional influence
    decay_rate: float = 0.95  # Rate at which influence decays with distance
    emotion_compatibility_factor: float = 0.5  # How similar emotions amplify influence
    crowd_amplification: bool = True  # Whether crowd effects amplify emotions
    contagion_threshold: float = 0.3  # Minimum emotion strength for contagion
    influence_memory_decay: float = 0.9  # How quickly influence memories fade
    leadership_factor: float = 1.5  # Multiplier for emotionally stable agents
    
    def __post_init__(self):
        """Validate contagion parameters."""
        self.base_influence_strength = max(0.0, min(1.0, self.base_influence_strength))
        self.max_influence_distance = max(10.0, self.max_influence_distance)
        self.decay_rate = max(0.1, min(1.0, self.decay_rate))
        self.emotion_compatibility_factor = max(0.0, min(2.0, self.emotion_compatibility_factor))
        self.contagion_threshold = max(0.0, min(1.0, self.contagion_threshold))
        self.influence_memory_decay = max(0.5, min(1.0, self.influence_memory_decay))
        self.leadership_factor = max(1.0, min(3.0, self.leadership_factor))


@dataclass
class EmotionalInfluence:
    """Represents emotional influence between agents."""
    
    source_agent_id: int
    target_agent_id: int
    influence_strength: float
    emotional_dimensions: Dict[str, float]
    distance: float
    timestamp: float
    emotion_type: EmotionType
    
    def is_expired(self, current_time: float, max_age: float = 10.0) -> bool:
        """Check if influence has expired."""
        return (current_time - self.timestamp) > max_age


class SentimentContagion:
    """
    Advanced sentiment contagion system for multi-agent environments.
    
    Implements realistic emotional spreading with distance-based decay,
    crowd effects, emotional compatibility, and leadership dynamics.
    """
    
    def __init__(self, parameters: Optional[ContagionParameters] = None):
        """
        Initialize sentiment contagion system.
        
        Args:
            parameters: Contagion parameters (uses defaults if None)
        """
        self.parameters = parameters or ContagionParameters()
        
        # Tracking structures
        self.active_influences: List[EmotionalInfluence] = []
        self.influence_network: Dict[int, List[int]] = defaultdict(list)
        self.emotional_clusters: Dict[str, List[int]] = defaultdict(list)
        
        # Performance metrics
        self.total_influences_processed = 0
        self.contagion_events = 0
        self.cluster_formations = 0
        
        logger.info("SentimentContagion system initialized")
    
    def process_emotional_contagion(self, agent_emotions: Dict[int, EmotionalState],
                                   agent_positions: Dict[int, np.ndarray]) -> Dict[int, List[EmotionalInfluence]]:
        """
        Process emotional contagion for all agents in the system.
        
        Args:
            agent_emotions: Dictionary of agent ID to emotional state
            agent_positions: Dictionary of agent ID to position
            
        Returns:
            Dictionary of agent ID to list of influences affecting them
        """
        try:
            current_time = time.time()
            influences_per_agent = defaultdict(list)
            
            # Clear expired influences
            self._cleanup_expired_influences(current_time)
            
            # Calculate pairwise influences
            agent_ids = list(agent_emotions.keys())
            
            for i, agent_id in enumerate(agent_ids):
                if agent_id not in agent_positions or agent_id not in agent_emotions:
                    continue
                    
                source_emotion = agent_emotions[agent_id]
                source_position = agent_positions[agent_id]
                
                # Check if agent can influence others (emotion strength threshold)
                dominant_emotion, emotion_strength = source_emotion.get_dominant_emotion()
                if emotion_strength < self.parameters.contagion_threshold:
                    continue
                
                # Find agents within influence radius
                for j, target_id in enumerate(agent_ids):
                    if i >= j or target_id not in agent_positions or target_id not in agent_emotions:
                        continue
                    
                    target_emotion = agent_emotions[target_id]
                    target_position = agent_positions[target_id]
                    
                    # Calculate distance and check if within influence range
                    distance = calculate_distance(source_position, target_position)
                    if distance > self.parameters.max_influence_distance:
                        continue
                    
                    # Calculate bidirectional influences
                    source_to_target = self._calculate_influence(
                        source_emotion, target_emotion, distance, current_time
                    )
                    
                    target_to_source = self._calculate_influence(
                        target_emotion, source_emotion, distance, current_time
                    )
                    
                    # Add significant influences
                    if source_to_target.influence_strength > 0.01:
                        influences_per_agent[target_id].append(source_to_target)
                        self.active_influences.append(source_to_target)
                    
                    if target_to_source.influence_strength > 0.01:
                        influences_per_agent[agent_id].append(target_to_source)
                        self.active_influences.append(target_to_source)
                    
                    self.total_influences_processed += 2
            
            # Apply crowd amplification effects
            if self.parameters.crowd_amplification:
                influences_per_agent = self._apply_crowd_effects(influences_per_agent, agent_emotions, agent_positions)
            
            # Update influence network for analysis
            self._update_influence_network(influences_per_agent)
            
            # Detect and track emotional clusters
            self._detect_emotional_clusters(agent_emotions, agent_positions)
            
            return dict(influences_per_agent)
            
        except Exception as e:
            logger.error(f"Emotional contagion processing failed: {str(e)}")
            return {}
    
    def apply_influences_to_agent(self, agent_emotion: EmotionalState,
                                influences: List[EmotionalInfluence]) -> None:
        """
        Apply emotional influences to a specific agent.
        
        Args:
            agent_emotion: Target agent's emotional state
            influences: List of influences affecting the agent
        """
        try:
            if not influences:
                return
            
            # Sort influences by strength (strongest first)
            influences.sort(key=lambda x: x.influence_strength, reverse=True)
            
            # Apply influences with diminishing returns
            total_arousal_influence = 0.0
            total_valence_influence = 0.0
            total_dominance_influence = 0.0
            
            influence_decay_factor = 1.0
            
            for influence in influences:
                # Apply decay for multiple influences
                effective_strength = influence.influence_strength * influence_decay_factor
                
                # Calculate dimensional influences
                arousal_diff = influence.emotional_dimensions.get('arousal', 0.0) - agent_emotion.arousal
                valence_diff = influence.emotional_dimensions.get('valence', 0.0) - agent_emotion.valence
                dominance_diff = influence.emotional_dimensions.get('dominance', 0.0) - agent_emotion.dominance
                
                # Apply influence with effective strength
                total_arousal_influence += arousal_diff * effective_strength
                total_valence_influence += valence_diff * effective_strength
                total_dominance_influence += dominance_diff * effective_strength
                
                # Diminishing returns for additional influences
                influence_decay_factor *= 0.8
                
                if influence_decay_factor < 0.1:
                    break  # Stop processing weak influences
            
            # Limit maximum influence per step to prevent instability
            max_influence = 0.2
            total_arousal_influence = max(-max_influence, min(max_influence, total_arousal_influence))
            total_valence_influence = max(-max_influence, min(max_influence, total_valence_influence))
            total_dominance_influence = max(-max_influence, min(max_influence, total_dominance_influence))
            
            # Apply influences to agent emotional state
            new_arousal = agent_emotion.arousal + total_arousal_influence
            new_valence = agent_emotion.valence + total_valence_influence
            new_dominance = agent_emotion.dominance + total_dominance_influence
            
            # Bound check
            agent_emotion.arousal = max(-1.0, min(1.0, new_arousal))
            agent_emotion.valence = max(-1.0, min(1.0, new_valence))
            agent_emotion.dominance = max(-1.0, min(1.0, new_dominance))
            
            # Record contagion event if significant change occurred
            total_change = abs(total_arousal_influence) + abs(total_valence_influence) + abs(total_dominance_influence)
            if total_change > 0.05:
                self.contagion_events += 1
            
        except Exception as e:
            logger.error(f"Applying influences to agent {agent_emotion.agent_id} failed: {str(e)}")
    
    def get_emotional_field_map(self, agent_emotions: Dict[int, EmotionalState],
                               agent_positions: Dict[int, np.ndarray],
                               grid_resolution: int = 50) -> Dict[str, np.ndarray]:
        """
        Generate emotional field maps for visualization and analysis.
        
        Args:
            agent_emotions: Dictionary of agent emotional states
            agent_positions: Dictionary of agent positions
            grid_resolution: Resolution of the generated field maps
            
        Returns:
            Dictionary containing arousal, valence, and dominance field maps
        """
        try:
            if not agent_positions:
                return {}
            
            # Determine field bounds
            positions = list(agent_positions.values())
            min_x = min(pos[0] for pos in positions)
            max_x = max(pos[0] for pos in positions)
            min_y = min(pos[1] for pos in positions)
            max_y = max(pos[1] for pos in positions)
            
            # Create coordinate grids
            x_coords = np.linspace(min_x, max_x, grid_resolution)
            y_coords = np.linspace(min_y, max_y, grid_resolution)
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # Initialize field maps
            arousal_field = np.zeros((grid_resolution, grid_resolution))
            valence_field = np.zeros((grid_resolution, grid_resolution))
            dominance_field = np.zeros((grid_resolution, grid_resolution))
            influence_strength_field = np.zeros((grid_resolution, grid_resolution))
            
            # Calculate emotional fields
            for i in range(grid_resolution):
                for j in range(grid_resolution):
                    grid_position = np.array([X[i, j], Y[i, j]])
                    
                    total_arousal = 0.0
                    total_valence = 0.0
                    total_dominance = 0.0
                    total_weight = 0.0
                    
                    # Sum weighted influences from all agents
                    for agent_id, agent_emotion in agent_emotions.items():
                        if agent_id not in agent_positions:
                            continue
                        
                        agent_position = agent_positions[agent_id]
                        distance = calculate_distance(grid_position, agent_position)
                        
                        # Calculate influence strength with distance decay
                        if distance < self.parameters.max_influence_distance:
                            weight = self._calculate_distance_weight(distance)
                            
                            # Get dominant emotion strength as multiplier
                            _, emotion_strength = agent_emotion.get_dominant_emotion()
                            weight *= emotion_strength
                            
                            total_arousal += agent_emotion.arousal * weight
                            total_valence += agent_emotion.valence * weight
                            total_dominance += agent_emotion.dominance * weight
                            total_weight += weight
                    
                    # Normalize by total weight
                    if total_weight > 0:
                        arousal_field[i, j] = total_arousal / total_weight
                        valence_field[i, j] = total_valence / total_weight
                        dominance_field[i, j] = total_dominance / total_weight
                        influence_strength_field[i, j] = min(1.0, total_weight)
            
            return {
                'arousal': arousal_field,
                'valence': valence_field,
                'dominance': dominance_field,
                'influence_strength': influence_strength_field,
                'x_coords': X,
                'y_coords': Y
            }
            
        except Exception as e:
            logger.error(f"Generating emotional field map failed: {str(e)}")
            return {}
    
    def get_contagion_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about emotional contagion processes.
        
        Returns:
            Dictionary containing contagion statistics
        """
        try:
            current_time = time.time()
            
            # Clean up expired influences for accurate counts
            self._cleanup_expired_influences(current_time)
            
            # Calculate network metrics
            network_size = len(self.influence_network)
            total_connections = sum(len(connections) for connections in self.influence_network.values())
            avg_connections = total_connections / max(1, network_size)
            
            # Analyze active influences
            influence_strengths = [inf.influence_strength for inf in self.active_influences]
            avg_influence_strength = np.mean(influence_strengths) if influence_strengths else 0.0
            max_influence_strength = max(influence_strengths) if influence_strengths else 0.0
            
            # Cluster statistics
            cluster_sizes = [len(agents) for agents in self.emotional_clusters.values()]
            avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0.0
            largest_cluster_size = max(cluster_sizes) if cluster_sizes else 0.0
            
            return {
                'total_influences_processed': self.total_influences_processed,
                'contagion_events': self.contagion_events,
                'cluster_formations': self.cluster_formations,
                'active_influences': len(self.active_influences),
                'network_size': network_size,
                'avg_connections_per_agent': avg_connections,
                'avg_influence_strength': avg_influence_strength,
                'max_influence_strength': max_influence_strength,
                'emotional_clusters': len(self.emotional_clusters),
                'avg_cluster_size': avg_cluster_size,
                'largest_cluster_size': largest_cluster_size
            }
            
        except Exception as e:
            logger.error(f"Getting contagion statistics failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_influence(self, source_emotion: EmotionalState, target_emotion: EmotionalState,
                           distance: float, current_time: float) -> EmotionalInfluence:
        """Calculate emotional influence between two agents."""
        try:
            # Base influence calculation
            source_dominant, source_strength = source_emotion.get_dominant_emotion()
            
            # Distance-based decay
            distance_weight = self._calculate_distance_weight(distance)
            
            # Emotional compatibility factor
            compatibility = self._calculate_emotional_compatibility(source_emotion, target_emotion)
            
            # Leadership factor based on emotional stability
            leadership_multiplier = 1.0
            if hasattr(source_emotion, 'emotional_stability'):
                if source_emotion.emotional_stability > 0.7:
                    leadership_multiplier = self.parameters.leadership_factor
            
            # Calculate final influence strength
            influence_strength = (
                self.parameters.base_influence_strength *
                source_strength *
                distance_weight *
                compatibility *
                leadership_multiplier
            )
            
            # Create influence object
            influence = EmotionalInfluence(
                source_agent_id=source_emotion.agent_id,
                target_agent_id=target_emotion.agent_id,
                influence_strength=influence_strength,
                emotional_dimensions={
                    'arousal': source_emotion.arousal,
                    'valence': source_emotion.valence,
                    'dominance': source_emotion.dominance
                },
                distance=distance,
                timestamp=current_time,
                emotion_type=source_dominant
            )
            
            return influence
            
        except Exception as e:
            logger.error(f"Calculating influence failed: {str(e)}")
            # Return neutral influence
            return EmotionalInfluence(
                source_agent_id=source_emotion.agent_id,
                target_agent_id=target_emotion.agent_id,
                influence_strength=0.0,
                emotional_dimensions={'arousal': 0.0, 'valence': 0.0, 'dominance': 0.0},
                distance=distance,
                timestamp=current_time,
                emotion_type=EmotionType.NEUTRAL
            )
    
    def _calculate_distance_weight(self, distance: float) -> float:
        """Calculate distance-based influence weight."""
        if distance >= self.parameters.max_influence_distance:
            return 0.0
        
        # Exponential decay with distance
        normalized_distance = distance / self.parameters.max_influence_distance
        weight = np.exp(-2.0 * normalized_distance) * self.parameters.decay_rate
        
        return max(0.0, min(1.0, weight))
    
    def _calculate_emotional_compatibility(self, source_emotion: EmotionalState,
                                         target_emotion: EmotionalState) -> float:
        """Calculate emotional compatibility between two agents."""
        try:
            # Calculate dimensional similarity
            arousal_diff = abs(source_emotion.arousal - target_emotion.arousal)
            valence_diff = abs(source_emotion.valence - target_emotion.valence)
            dominance_diff = abs(source_emotion.dominance - target_emotion.dominance)
            
            # Average difference (0 = identical, 2 = maximally different)
            avg_diff = (arousal_diff + valence_diff + dominance_diff) / 3.0
            
            # Convert to compatibility (1 = identical, 0 = maximally different)
            base_compatibility = max(0.0, 1.0 - (avg_diff / 2.0))
            
            # Apply compatibility factor
            compatibility = 1.0 + (base_compatibility - 0.5) * self.parameters.emotion_compatibility_factor
            
            return max(0.1, min(2.0, compatibility))
            
        except Exception as e:
            logger.warning(f"Emotional compatibility calculation failed: {str(e)}")
            return 1.0  # Neutral compatibility
    
    def _apply_crowd_effects(self, influences_per_agent: Dict[int, List[EmotionalInfluence]],
                           agent_emotions: Dict[int, EmotionalState],
                           agent_positions: Dict[int, np.ndarray]) -> Dict[int, List[EmotionalInfluence]]:
        """Apply crowd amplification effects to emotional influences."""
        try:
            crowd_radius = self.parameters.max_influence_distance * 0.5
            
            for agent_id in influences_per_agent:
                if agent_id not in agent_positions:
                    continue
                
                agent_position = agent_positions[agent_id]
                
                # Count nearby agents
                nearby_count = 0
                for other_id, other_position in agent_positions.items():
                    if other_id != agent_id:
                        distance = calculate_distance(agent_position, other_position)
                        if distance < crowd_radius:
                            nearby_count += 1
                
                # Apply crowd amplification
                if nearby_count > 3:  # Crowd threshold
                    crowd_factor = 1.0 + min(0.5, (nearby_count - 3) * 0.1)
                    
                    # Amplify all influences
                    for influence in influences_per_agent[agent_id]:
                        influence.influence_strength *= crowd_factor
                        influence.influence_strength = min(1.0, influence.influence_strength)
            
            return influences_per_agent
            
        except Exception as e:
            logger.error(f"Applying crowd effects failed: {str(e)}")
            return influences_per_agent
    
    def _update_influence_network(self, influences_per_agent: Dict[int, List[EmotionalInfluence]]) -> None:
        """Update influence network for analysis."""
        try:
            # Clear old network
            self.influence_network.clear()
            
            # Build new network from active influences
            for target_id, influences in influences_per_agent.items():
                for influence in influences:
                    source_id = influence.source_agent_id
                    if source_id not in self.influence_network:
                        self.influence_network[source_id] = []
                    
                    if target_id not in self.influence_network[source_id]:
                        self.influence_network[source_id].append(target_id)
            
        except Exception as e:
            logger.warning(f"Updating influence network failed: {str(e)}")
    
    def _detect_emotional_clusters(self, agent_emotions: Dict[int, EmotionalState],
                                 agent_positions: Dict[int, np.ndarray]) -> None:
        """Detect and track emotional clusters in the population."""
        try:
            # Clear old clusters
            old_cluster_count = len(self.emotional_clusters)
            self.emotional_clusters.clear()
            
            # Group agents by dominant emotion
            emotion_groups = defaultdict(list)
            
            for agent_id, emotion in agent_emotions.items():
                if agent_id not in agent_positions:
                    continue
                
                dominant_emotion, strength = emotion.get_dominant_emotion()
                if strength > self.parameters.contagion_threshold:
                    emotion_groups[dominant_emotion.value].append(agent_id)
            
            # Find spatially coherent clusters within each emotion group
            cluster_radius = self.parameters.max_influence_distance * 0.7
            
            for emotion_type, agent_ids in emotion_groups.items():
                if len(agent_ids) < 3:  # Minimum cluster size
                    continue
                
                # Use simple spatial clustering
                unassigned = set(agent_ids)
                cluster_id = 0
                
                while unassigned:
                    # Start new cluster with arbitrary agent
                    seed_agent = next(iter(unassigned))
                    cluster = {seed_agent}
                    unassigned.remove(seed_agent)
                    seed_position = agent_positions[seed_agent]
                    
                    # Find nearby agents with same emotion
                    for other_agent in list(unassigned):
                        other_position = agent_positions[other_agent]
                        distance = calculate_distance(seed_position, other_position)
                        
                        if distance < cluster_radius:
                            cluster.add(other_agent)
                            unassigned.remove(other_agent)
                    
                    # Only keep clusters with minimum size
                    if len(cluster) >= 3:
                        cluster_key = f"{emotion_type}_cluster_{cluster_id}"
                        self.emotional_clusters[cluster_key] = list(cluster)
                        cluster_id += 1
            
            # Track cluster formation events
            new_cluster_count = len(self.emotional_clusters)
            if new_cluster_count > old_cluster_count:
                self.cluster_formations += (new_cluster_count - old_cluster_count)
            
        except Exception as e:
            logger.warning(f"Detecting emotional clusters failed: {str(e)}")
    
    def _cleanup_expired_influences(self, current_time: float) -> None:
        """Remove expired influences from active list."""
        try:
            initial_count = len(self.active_influences)
            self.active_influences = [
                influence for influence in self.active_influences
                if not influence.is_expired(current_time)
            ]
            
            removed_count = initial_count - len(self.active_influences)
            if removed_count > 0:
                logger.debug(f"Cleaned up {removed_count} expired influences")
                
        except Exception as e:
            logger.warning(f"Cleaning up expired influences failed: {str(e)}")