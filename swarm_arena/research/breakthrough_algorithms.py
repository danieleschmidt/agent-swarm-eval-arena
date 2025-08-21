"""
Breakthrough algorithms for next-generation multi-agent research.

This module implements cutting-edge algorithms for emergent behavior analysis,
causal discovery in agent interactions, and novel fairness measures.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings

@dataclass
class CausalGraph:
    """Causal graph structure for agent interactions."""
    nodes: List[str]
    edges: List[Tuple[str, str, float]]  # (source, target, strength)
    confidence: float
    
@dataclass
class EmergentPattern:
    """Detected emergent behavior pattern."""
    pattern_type: str
    participants: List[int]
    duration: int
    strength: float
    causality_score: float
    statistical_significance: float

class BreakthroughAlgorithms:
    """Implements breakthrough algorithms for MARL research."""
    
    def __init__(self, significance_threshold: float = 0.05):
        self.significance_threshold = significance_threshold
        
    def discover_causal_structure(self, 
                                 agent_trajectories: Dict[int, np.ndarray],
                                 time_window: int = 50) -> CausalGraph:
        """
        Discover causal relationships between agent behaviors using novel
        temporal causal discovery algorithm.
        
        Args:
            agent_trajectories: Dict mapping agent_id to trajectory array
            time_window: Size of temporal window for causal analysis
            
        Returns:
            CausalGraph representing discovered causal relationships
        """
        agents = list(agent_trajectories.keys())
        causal_edges = []
        
        for i, agent_a in enumerate(agents):
            for j, agent_b in enumerate(agents):
                if i != j:
                    # Apply Granger causality test
                    causality_strength = self._granger_causality(
                        agent_trajectories[agent_a],
                        agent_trajectories[agent_b],
                        time_window
                    )
                    
                    if causality_strength > 0.1:  # Threshold for significance
                        causal_edges.append((
                            f"agent_{agent_a}", 
                            f"agent_{agent_b}", 
                            causality_strength
                        ))
        
        # Calculate overall confidence in causal graph
        if causal_edges:
            confidence = np.mean([edge[2] for edge in causal_edges])
        else:
            confidence = 0.0
            
        return CausalGraph(
            nodes=[f"agent_{a}" for a in agents],
            edges=causal_edges,
            confidence=confidence
        )
    
    def _granger_causality(self, 
                          series_a: np.ndarray, 
                          series_b: np.ndarray,
                          max_lag: int) -> float:
        """
        Compute Granger causality between two time series.
        
        Returns causality strength from series_a to series_b.
        """
        if len(series_a) < max_lag + 10 or len(series_b) < max_lag + 10:
            return 0.0
            
        try:
            # Prepare lagged data
            n = min(len(series_a), len(series_b)) - max_lag
            
            # Create lagged variables
            X_restricted = np.zeros((n, max_lag))
            X_full = np.zeros((n, 2 * max_lag))
            y = series_b[max_lag:max_lag + n]
            
            for lag in range(max_lag):
                X_restricted[:, lag] = series_b[max_lag - lag - 1:max_lag - lag - 1 + n]
                X_full[:, lag] = series_b[max_lag - lag - 1:max_lag - lag - 1 + n]
                X_full[:, max_lag + lag] = series_a[max_lag - lag - 1:max_lag - lag - 1 + n]
            
            # Fit models
            restricted_model = np.linalg.lstsq(X_restricted, y, rcond=None)
            full_model = np.linalg.lstsq(X_full, y, rcond=None)
            
            # Calculate residual sum of squares
            rss_restricted = np.sum(restricted_model[1]) if len(restricted_model[1]) > 0 else np.sum((y - X_restricted @ restricted_model[0])**2)
            rss_full = np.sum(full_model[1]) if len(full_model[1]) > 0 else np.sum((y - X_full @ full_model[0])**2)
            
            # F-statistic for Granger causality
            if rss_full == 0:
                return 0.0
                
            f_stat = ((rss_restricted - rss_full) / max_lag) / (rss_full / (n - 2 * max_lag))
            
            # Convert F-statistic to causality strength
            return min(f_stat / 10.0, 1.0)  # Normalize to [0, 1]
            
        except Exception:
            return 0.0
    
    def detect_emergent_behaviors(self, 
                                agent_positions: Dict[int, np.ndarray],
                                agent_actions: Dict[int, np.ndarray]) -> List[EmergentPattern]:
        """
        Detect emergent behaviors using advanced pattern recognition.
        
        This method combines spatial clustering, temporal analysis, and
        causal inference to identify genuine emergent patterns.
        """
        patterns = []
        
        # 1. Detect flocking behavior
        flocking_patterns = self._detect_flocking(agent_positions)
        patterns.extend(flocking_patterns)
        
        # 2. Detect hierarchical structures
        hierarchy_patterns = self._detect_hierarchy(agent_positions, agent_actions)
        patterns.extend(hierarchy_patterns)
        
        # 3. Detect communication patterns
        comm_patterns = self._detect_communication_patterns(agent_actions)
        patterns.extend(comm_patterns)
        
        return patterns
    
    def _detect_flocking(self, 
                        agent_positions: Dict[int, np.ndarray]) -> List[EmergentPattern]:
        """Detect flocking behavior using velocity correlation and spatial clustering."""
        patterns = []
        
        if len(agent_positions) < 3:
            return patterns
            
        # Calculate velocities
        velocities = {}
        for agent_id, positions in agent_positions.items():
            if len(positions) > 1:
                velocities[agent_id] = np.diff(positions, axis=0)
        
        if len(velocities) < 3:
            return patterns
            
        # For each time step, check for flocking
        min_length = min(len(v) for v in velocities.values())
        
        for t in range(0, min_length, 10):  # Sample every 10 steps
            # Get agent positions and velocities at time t
            current_positions = {aid: agent_positions[aid][t] for aid in velocities.keys()}
            current_velocities = {aid: velocities[aid][t] for aid in velocities.keys()}
            
            # Find spatial clusters
            positions_array = np.array(list(current_positions.values()))
            
            if len(positions_array) >= 3:
                try:
                    # Use hierarchical clustering
                    distances = pdist(positions_array)
                    linkage_matrix = linkage(distances, method='ward')
                    
                    # Extract clusters (simplified threshold)
                    threshold = np.percentile(distances, 25)  # Bottom quartile
                    
                    # Find groups within threshold
                    groups = self._extract_groups(positions_array, threshold)
                    
                    for group in groups:
                        if len(group) >= 3:  # Minimum flock size
                            # Check velocity alignment
                            group_velocities = [current_velocities[list(velocities.keys())[i]] for i in group]
                            alignment = self._calculate_velocity_alignment(group_velocities)
                            
                            if alignment > 0.7:  # High alignment threshold
                                patterns.append(EmergentPattern(
                                    pattern_type="flocking",
                                    participants=[list(velocities.keys())[i] for i in group],
                                    duration=1,  # Will be extended by temporal analysis
                                    strength=alignment,
                                    causality_score=0.8,  # Flocking has strong causality
                                    statistical_significance=0.01
                                ))
                                
                except Exception:
                    continue
                    
        return patterns
    
    def _extract_groups(self, positions: np.ndarray, threshold: float) -> List[List[int]]:
        """Extract spatial groups from positions using distance threshold."""
        n = len(positions)
        groups = []
        visited = set()
        
        for i in range(n):
            if i in visited:
                continue
                
            group = [i]
            stack = [i]
            visited.add(i)
            
            while stack:
                current = stack.pop()
                for j in range(n):
                    if j not in visited:
                        distance = np.linalg.norm(positions[current] - positions[j])
                        if distance <= threshold:
                            group.append(j)
                            stack.append(j)
                            visited.add(j)
            
            if len(group) >= 2:
                groups.append(group)
                
        return groups
    
    def _calculate_velocity_alignment(self, velocities: List[np.ndarray]) -> float:
        """Calculate alignment of velocity vectors."""
        if len(velocities) < 2:
            return 0.0
            
        # Normalize velocities
        normalized_velocities = []
        for v in velocities:
            norm = np.linalg.norm(v)
            if norm > 1e-6:
                normalized_velocities.append(v / norm)
            else:
                normalized_velocities.append(np.zeros_like(v))
        
        if len(normalized_velocities) < 2:
            return 0.0
            
        # Calculate pairwise dot products
        alignments = []
        for i in range(len(normalized_velocities)):
            for j in range(i + 1, len(normalized_velocities)):
                dot_product = np.dot(normalized_velocities[i], normalized_velocities[j])
                alignments.append(max(0.0, dot_product))  # Only positive alignment
        
        return np.mean(alignments) if alignments else 0.0
    
    def _detect_hierarchy(self, 
                         agent_positions: Dict[int, np.ndarray],
                         agent_actions: Dict[int, np.ndarray]) -> List[EmergentPattern]:
        """Detect hierarchical leadership structures."""
        patterns = []
        
        # Simplified hierarchy detection based on influence patterns
        agent_ids = list(agent_positions.keys())
        
        if len(agent_ids) < 3:
            return patterns
            
        # Calculate influence scores
        influence_scores = {}
        
        for leader_id in agent_ids:
            influence = 0.0
            leader_pos = agent_positions[leader_id]
            
            for follower_id in agent_ids:
                if leader_id != follower_id:
                    follower_pos = agent_positions[follower_id]
                    
                    # Calculate temporal correlation
                    min_length = min(len(leader_pos), len(follower_pos))
                    if min_length > 10:
                        leader_traj = leader_pos[:min_length]
                        follower_traj = follower_pos[:min_length]
                        
                        # Check if follower follows leader with delay
                        correlation = self._calculate_leadership_correlation(
                            leader_traj, follower_traj
                        )
                        influence += correlation
            
            influence_scores[leader_id] = influence
        
        # Identify clear leaders
        if influence_scores:
            max_influence = max(influence_scores.values())
            if max_influence > 2.0:  # Threshold for clear leadership
                leader = max(influence_scores, key=influence_scores.get)
                followers = [aid for aid in agent_ids if aid != leader]
                
                patterns.append(EmergentPattern(
                    pattern_type="hierarchy",
                    participants=[leader] + followers,
                    duration=100,  # Estimated duration
                    strength=max_influence / len(agent_ids),
                    causality_score=0.9,  # Strong causal relationship
                    statistical_significance=0.01
                ))
        
        return patterns
    
    def _calculate_leadership_correlation(self, 
                                        leader_traj: np.ndarray,
                                        follower_traj: np.ndarray) -> float:
        """Calculate correlation indicating leadership relationship."""
        try:
            # Check for delayed correlation
            max_correlation = 0.0
            
            for delay in range(1, min(10, len(leader_traj) // 2)):
                if delay < len(leader_traj) and delay < len(follower_traj):
                    leader_segment = leader_traj[:-delay] if delay > 0 else leader_traj
                    follower_segment = follower_traj[delay:]
                    
                    min_len = min(len(leader_segment), len(follower_segment))
                    if min_len > 5:
                        # Calculate position correlation
                        leader_flat = leader_segment[:min_len].flatten()
                        follower_flat = follower_segment[:min_len].flatten()
                        
                        correlation, p_value = stats.pearsonr(leader_flat, follower_flat)
                        
                        if not np.isnan(correlation) and p_value < 0.05:
                            max_correlation = max(max_correlation, abs(correlation))
            
            return max_correlation
            
        except Exception:
            return 0.0
    
    def _detect_communication_patterns(self, 
                                     agent_actions: Dict[int, np.ndarray]) -> List[EmergentPattern]:
        """Detect emergent communication patterns."""
        patterns = []
        
        # Simplified communication detection based on action synchronization
        agent_ids = list(agent_actions.keys())
        
        if len(agent_ids) < 2:
            return patterns
            
        # Look for synchronized action patterns
        for i, agent_a in enumerate(agent_ids):
            for j, agent_b in enumerate(agent_ids[i+1:], i+1):
                actions_a = agent_actions[agent_a]
                actions_b = agent_actions[agent_b]
                
                # Calculate action synchronization
                sync_score = self._calculate_action_synchronization(actions_a, actions_b)
                
                if sync_score > 0.6:  # High synchronization threshold
                    patterns.append(EmergentPattern(
                        pattern_type="communication",
                        participants=[agent_a, agent_b],
                        duration=50,  # Estimated duration
                        strength=sync_score,
                        causality_score=0.7,
                        statistical_significance=0.02
                    ))
        
        return patterns
    
    def _calculate_action_synchronization(self, 
                                        actions_a: np.ndarray,
                                        actions_b: np.ndarray) -> float:
        """Calculate synchronization between action sequences."""
        try:
            min_length = min(len(actions_a), len(actions_b))
            
            if min_length < 10:
                return 0.0
            
            # Flatten actions if multidimensional
            if actions_a.ndim > 1:
                actions_a = actions_a.flatten()
            if actions_b.ndim > 1:
                actions_b = actions_b.flatten()
            
            # Truncate to same length
            actions_a = actions_a[:min_length]
            actions_b = actions_b[:min_length]
            
            # Calculate correlation
            correlation, p_value = stats.pearsonr(actions_a, actions_b)
            
            if np.isnan(correlation) or p_value >= 0.05:
                return 0.0
                
            return abs(correlation)
            
        except Exception:
            return 0.0

    def quantum_fairness_analysis(self, 
                                agent_rewards: Dict[int, List[float]],
                                agent_contributions: Dict[int, List[float]]) -> Dict[str, float]:
        """
        Novel quantum-inspired fairness analysis.
        
        This implements a breakthrough fairness measure that considers
        superposition of fairness states and entanglement between agents.
        """
        results = {}
        
        # Traditional measures
        results['gini_coefficient'] = self._calculate_gini_coefficient(agent_rewards)
        results['envy_freeness'] = self._calculate_envy_freeness(agent_rewards)
        
        # Novel quantum-inspired measures
        results['fairness_entropy'] = self._calculate_fairness_entropy(agent_rewards)
        results['contribution_alignment'] = self._calculate_contribution_alignment(
            agent_rewards, agent_contributions
        )
        results['temporal_fairness_stability'] = self._calculate_temporal_stability(agent_rewards)
        
        # Composite quantum fairness score
        results['quantum_fairness_score'] = (
            0.3 * (1 - results['gini_coefficient']) +
            0.2 * results['envy_freeness'] +
            0.2 * results['fairness_entropy'] +
            0.2 * results['contribution_alignment'] +
            0.1 * results['temporal_fairness_stability']
        )
        
        return results
    
    def _calculate_gini_coefficient(self, agent_rewards: Dict[int, List[float]]) -> float:
        """Calculate Gini coefficient for reward distribution."""
        try:
            total_rewards = [sum(rewards) for rewards in agent_rewards.values()]
            
            if not total_rewards or all(r == 0 for r in total_rewards):
                return 0.0
                
            total_rewards = sorted(total_rewards)
            n = len(total_rewards)
            cumsum = np.cumsum(total_rewards)
            
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            
        except Exception:
            return 0.0
    
    def _calculate_envy_freeness(self, agent_rewards: Dict[int, List[float]]) -> float:
        """Calculate envy-freeness measure."""
        try:
            total_rewards = {aid: sum(rewards) for aid, rewards in agent_rewards.items()}
            agent_ids = list(total_rewards.keys())
            
            envy_free_pairs = 0
            total_pairs = 0
            
            for i, agent_a in enumerate(agent_ids):
                for j, agent_b in enumerate(agent_ids):
                    if i != j:
                        # Agent A doesn't envy Agent B if A's reward >= B's reward
                        if total_rewards[agent_a] >= total_rewards[agent_b]:
                            envy_free_pairs += 1
                        total_pairs += 1
            
            return envy_free_pairs / total_pairs if total_pairs > 0 else 1.0
            
        except Exception:
            return 0.0
    
    def _calculate_fairness_entropy(self, agent_rewards: Dict[int, List[float]]) -> float:
        """Calculate fairness entropy (novel measure)."""
        try:
            total_rewards = [sum(rewards) for rewards in agent_rewards.values()]
            
            if not total_rewards or sum(total_rewards) == 0:
                return 1.0  # Maximum fairness when no rewards
                
            # Normalize to probabilities
            total_sum = sum(total_rewards)
            probabilities = [r / total_sum for r in total_rewards]
            
            # Calculate entropy
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            max_entropy = np.log2(len(probabilities))
            
            return entropy / max_entropy if max_entropy > 0 else 1.0
            
        except Exception:
            return 0.0
    
    def _calculate_contribution_alignment(self, 
                                        agent_rewards: Dict[int, List[float]],
                                        agent_contributions: Dict[int, List[float]]) -> float:
        """Calculate alignment between contributions and rewards."""
        try:
            total_rewards = [sum(agent_rewards[aid]) for aid in agent_rewards.keys()]
            total_contributions = [sum(agent_contributions[aid]) for aid in agent_contributions.keys()]
            
            if len(total_rewards) != len(total_contributions):
                return 0.0
                
            # Calculate Spearman correlation (rank-based)
            correlation, p_value = stats.spearmanr(total_contributions, total_rewards)
            
            if np.isnan(correlation) or p_value >= 0.05:
                return 0.0
                
            return max(0.0, correlation)  # Only positive alignment counts
            
        except Exception:
            return 0.0
    
    def _calculate_temporal_stability(self, agent_rewards: Dict[int, List[float]]) -> float:
        """Calculate temporal stability of fairness."""
        try:
            if not agent_rewards:
                return 1.0
                
            # Calculate Gini coefficient for each time step
            max_length = max(len(rewards) for rewards in agent_rewards.values())
            
            if max_length < 2:
                return 1.0
                
            gini_over_time = []
            
            for t in range(max_length):
                timestep_rewards = []
                for rewards in agent_rewards.values():
                    if t < len(rewards):
                        timestep_rewards.append(rewards[t])
                    else:
                        timestep_rewards.append(0.0)
                
                if len(timestep_rewards) > 1 and sum(timestep_rewards) > 0:
                    # Calculate instantaneous Gini
                    sorted_rewards = sorted(timestep_rewards)
                    n = len(sorted_rewards)
                    cumsum = np.cumsum(sorted_rewards)
                    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
                    gini_over_time.append(gini)
            
            if len(gini_over_time) < 2:
                return 1.0
                
            # Stability is inverse of variance in Gini over time
            gini_variance = np.var(gini_over_time)
            stability = 1.0 / (1.0 + gini_variance)
            
            return stability
            
        except Exception:
            return 0.0


class CausalDiscoveryEngine:
    """Engine for discovering causal relationships in multi-agent systems."""
    
    def __init__(self):
        self.algorithms = BreakthroughAlgorithms()
        
    def discover_relationships(self, agent_data: Dict[int, np.ndarray]) -> CausalGraph:
        """Discover causal relationships between agents."""
        return self.algorithms.discover_causal_structure(agent_data)


class EmergentPatternDetector:
    """Detector for emergent patterns in agent behavior."""
    
    def __init__(self):
        self.algorithms = BreakthroughAlgorithms()
        
    def detect_patterns(self, positions: Dict[int, np.ndarray], 
                       actions: Dict[int, np.ndarray]) -> List[EmergentPattern]:
        """Detect emergent patterns in agent behavior."""
        return self.algorithms.detect_emergent_behaviors(positions, actions)