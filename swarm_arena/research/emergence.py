"""Emergence detection and pattern analysis for swarm behaviors."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import networkx as nx


@dataclass
class EmergentPattern:
    """Represents a detected emergent pattern."""
    
    name: str
    confidence: float
    agents: List[int]
    duration: int
    start_time: int
    end_time: int
    pattern_type: str
    metadata: Dict[str, Any]
    
    def visualize(self, save_path: Optional[str] = None) -> None:
        """Visualize the emergent pattern."""
        # Placeholder for visualization logic
        print(f"Visualizing pattern: {self.name}")
        if save_path:
            print(f"Would save to: {save_path}")


class EmergenceDetector:
    """Detects emergent behaviors in swarm trajectories."""
    
    def __init__(self, 
                 min_pattern_duration: int = 10,
                 confidence_threshold: float = 0.7):
        """Initialize emergence detector.
        
        Args:
            min_pattern_duration: Minimum duration for a pattern to be considered emergent
            confidence_threshold: Minimum confidence for pattern detection
        """
        self.min_pattern_duration = min_pattern_duration
        self.confidence_threshold = confidence_threshold
        
    def analyze(self, trajectories: Dict[int, List[Tuple[float, float]]]) -> List[EmergentPattern]:
        """Analyze trajectories for emergent patterns.
        
        Args:
            trajectories: Dict mapping agent_id to list of (x, y) positions over time
            
        Returns:
            List of detected emergent patterns
        """
        patterns = []
        
        # Convert trajectories to numpy arrays for analysis
        agent_positions = self._prepare_trajectory_data(trajectories)
        
        # Detect different types of patterns
        patterns.extend(self._detect_flocking(agent_positions))
        patterns.extend(self._detect_clustering(agent_positions))
        patterns.extend(self._detect_leadership(agent_positions))
        patterns.extend(self._detect_oscillations(agent_positions))
        
        # Filter by confidence and duration
        filtered_patterns = [
            p for p in patterns 
            if p.confidence >= self.confidence_threshold and 
               p.duration >= self.min_pattern_duration
        ]
        
        return filtered_patterns
    
    def _prepare_trajectory_data(self, trajectories: Dict[int, List[Tuple[float, float]]]) -> np.ndarray:
        """Convert trajectory data to numpy array format.
        
        Args:
            trajectories: Raw trajectory data
            
        Returns:
            Array of shape (num_agents, max_time_steps, 2) with positions
        """
        if not trajectories:
            return np.array([])
        
        max_time_steps = max(len(traj) for traj in trajectories.values())
        num_agents = len(trajectories)
        
        # Initialize with NaN for missing data
        positions = np.full((num_agents, max_time_steps, 2), np.nan)
        
        for i, (agent_id, traj) in enumerate(trajectories.items()):
            for t, (x, y) in enumerate(traj):
                positions[i, t, 0] = x
                positions[i, t, 1] = y
        
        return positions
    
    def _detect_flocking(self, agent_positions: np.ndarray) -> List[EmergentPattern]:
        """Detect flocking behavior using velocity alignment."""
        patterns = []
        
        if agent_positions.size == 0:
            return patterns
        
        num_agents, num_steps, _ = agent_positions.shape
        
        if num_steps < 2:
            return patterns
        
        # Calculate velocities
        velocities = np.diff(agent_positions, axis=1)
        
        # For each time window, check for velocity alignment
        window_size = max(10, self.min_pattern_duration)
        
        for start_t in range(0, num_steps - window_size, window_size // 2):
            end_t = start_t + window_size
            
            # Calculate alignment metric for this window
            window_velocities = velocities[:, start_t:end_t, :]
            
            # Skip if too many NaN values
            if np.isnan(window_velocities).sum() > window_velocities.size * 0.5:
                continue
            
            alignment_score = self._calculate_velocity_alignment(window_velocities)
            
            if alignment_score > 0.8:  # High alignment threshold
                pattern = EmergentPattern(
                    name="flocking",
                    confidence=alignment_score,
                    agents=list(range(num_agents)),
                    duration=window_size,
                    start_time=start_t,
                    end_time=end_t,
                    pattern_type="movement",
                    metadata={"alignment_score": alignment_score}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_clustering(self, agent_positions: np.ndarray) -> List[EmergentPattern]:
        """Detect spatial clustering using DBSCAN."""
        patterns = []
        
        if agent_positions.size == 0:
            return patterns
        
        num_agents, num_steps, _ = agent_positions.shape
        
        # Analyze clustering at regular intervals
        for t in range(0, num_steps, max(1, num_steps // 20)):
            positions_t = agent_positions[:, t, :]
            
            # Skip if too many missing positions
            valid_mask = ~np.isnan(positions_t).any(axis=1)
            if valid_mask.sum() < 3:
                continue
            
            valid_positions = positions_t[valid_mask]
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=50, min_samples=3).fit(valid_positions)
            
            # Check for meaningful clusters
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            
            if n_clusters >= 2:
                # Calculate clustering quality
                if n_clusters > 1 and len(set(clustering.labels_)) > 1:
                    try:
                        silhouette_avg = silhouette_score(valid_positions, clustering.labels_)
                        
                        if silhouette_avg > 0.3:  # Reasonable clustering threshold
                            agent_ids = np.where(valid_mask)[0].tolist()
                            
                            pattern = EmergentPattern(
                                name="spatial_clustering",
                                confidence=silhouette_avg,
                                agents=agent_ids,
                                duration=1,  # Snapshot pattern
                                start_time=t,
                                end_time=t + 1,
                                pattern_type="spatial",
                                metadata={
                                    "n_clusters": n_clusters,
                                    "silhouette_score": silhouette_avg
                                }
                            )
                            patterns.append(pattern)
                    except ValueError:
                        # Skip if silhouette score can't be calculated
                        continue
        
        return patterns
    
    def _detect_leadership(self, agent_positions: np.ndarray) -> List[EmergentPattern]:
        """Detect leadership patterns based on movement initiation."""
        patterns = []
        
        if agent_positions.size == 0:
            return patterns
        
        num_agents, num_steps, _ = agent_positions.shape
        
        if num_steps < 5:
            return patterns
        
        # Calculate movement initiation scores
        velocities = np.diff(agent_positions, axis=1)
        speeds = np.linalg.norm(velocities, axis=2)
        
        # For each agent, count how often they move first
        leadership_scores = np.zeros(num_agents)
        
        for t in range(num_steps - 2):
            # Find which agents start moving at this timestep
            current_speeds = speeds[:, t]
            next_speeds = speeds[:, t + 1]
            
            # Agent is a leader if they accelerate while others are slow
            leaders = (current_speeds < 1.0) & (next_speeds > 2.0)
            followers = (current_speeds < 1.0) & (next_speeds > 1.0) & ~leaders
            
            if leaders.any() and followers.any():
                # Award leadership points
                for leader_idx in np.where(leaders)[0]:
                    leadership_scores[leader_idx] += 1
        
        # Identify consistent leaders
        max_leadership = leadership_scores.max()
        if max_leadership > self.min_pattern_duration:
            leader_agents = np.where(leadership_scores > max_leadership * 0.7)[0].tolist()
            
            pattern = EmergentPattern(
                name="leadership",
                confidence=min(1.0, max_leadership / (num_steps * 0.1)),
                agents=leader_agents,
                duration=num_steps,
                start_time=0,
                end_time=num_steps,
                pattern_type="behavioral",
                metadata={"leadership_scores": leadership_scores.tolist()}
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_oscillations(self, agent_positions: np.ndarray) -> List[EmergentPattern]:
        """Detect oscillatory movement patterns."""
        patterns = []
        
        if agent_positions.size == 0:
            return patterns
        
        num_agents, num_steps, _ = agent_positions.shape
        
        if num_steps < 20:
            return patterns
        
        # Analyze center of mass oscillations
        center_of_mass = np.nanmean(agent_positions, axis=0)
        
        # Look for periodic behavior in center of mass
        for dim in range(2):  # x and y dimensions
            signal = center_of_mass[:, dim]
            
            # Remove NaN values
            valid_mask = ~np.isnan(signal)
            if valid_mask.sum() < 10:
                continue
            
            valid_signal = signal[valid_mask]
            
            # Simple oscillation detection using autocorrelation
            autocorr = np.correlate(valid_signal, valid_signal, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Look for peaks in autocorrelation
            if len(autocorr) > 10:
                # Find period with highest correlation after delay
                period_range = range(5, min(len(autocorr) // 2, 50))
                if period_range:
                    max_corr = max(autocorr[p] for p in period_range)
                    
                    if max_corr > 0.7:  # Strong oscillation
                        pattern = EmergentPattern(
                            name="oscillation",
                            confidence=max_corr,
                            agents=list(range(num_agents)),
                            duration=num_steps,
                            start_time=0,
                            end_time=num_steps,
                            pattern_type="temporal",
                            metadata={
                                "dimension": "x" if dim == 0 else "y",
                                "max_correlation": max_corr
                            }
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _calculate_velocity_alignment(self, velocities: np.ndarray) -> float:
        """Calculate velocity alignment score for a group of agents.
        
        Args:
            velocities: Array of shape (num_agents, time_steps, 2)
            
        Returns:
            Alignment score between 0 and 1
        """
        if velocities.size == 0:
            return 0.0
        
        # Calculate unit vectors for each velocity
        speeds = np.linalg.norm(velocities, axis=2, keepdims=True)
        speeds = np.where(speeds == 0, 1, speeds)  # Avoid division by zero
        
        unit_velocities = velocities / speeds
        
        # Calculate pairwise alignment and average
        alignments = []
        
        for t in range(velocities.shape[1]):
            vel_t = unit_velocities[:, t, :]
            
            # Skip timesteps with too many NaN values
            valid_mask = ~np.isnan(vel_t).any(axis=1)
            if valid_mask.sum() < 2:
                continue
            
            valid_vel = vel_t[valid_mask]
            
            # Calculate mean direction
            mean_direction = np.mean(valid_vel, axis=0)
            mean_direction_norm = np.linalg.norm(mean_direction)
            
            if mean_direction_norm > 0:
                # Alignment is the magnitude of the mean unit vector
                alignments.append(mean_direction_norm)
        
        return np.mean(alignments) if alignments else 0.0


class PatternAnalyzer:
    """Advanced pattern analysis tools."""
    
    def __init__(self):
        self.detector = EmergenceDetector()
    
    def analyze_swarm_dynamics(self, trajectories: Dict[int, List[Tuple[float, float]]]) -> Dict[str, Any]:
        """Comprehensive analysis of swarm dynamics.
        
        Args:
            trajectories: Agent trajectory data
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "patterns": [],
            "network_metrics": {},
            "temporal_analysis": {},
            "spatial_analysis": {}
        }
        
        # Detect emergent patterns
        patterns = self.detector.analyze(trajectories)
        results["patterns"] = [
            {
                "name": p.name,
                "confidence": p.confidence,
                "duration": p.duration,
                "agents_involved": len(p.agents),
                "type": p.pattern_type
            }
            for p in patterns
        ]
        
        # Network analysis
        results["network_metrics"] = self._analyze_interaction_network(trajectories)
        
        # Temporal analysis
        results["temporal_analysis"] = self._analyze_temporal_dynamics(trajectories)
        
        # Spatial analysis
        results["spatial_analysis"] = self._analyze_spatial_distribution(trajectories)
        
        return results
    
    def _analyze_interaction_network(self, trajectories: Dict[int, List[Tuple[float, float]]]) -> Dict[str, Any]:
        """Analyze agent interaction networks."""
        if not trajectories:
            return {}
        
        # Build interaction graph based on proximity
        G = nx.Graph()
        agent_ids = list(trajectories.keys())
        G.add_nodes_from(agent_ids)
        
        # Add edges between agents that are frequently close
        interaction_threshold = 50.0  # Distance threshold for interactions
        min_interactions = 10  # Minimum number of close encounters
        
        for i, agent1 in enumerate(agent_ids):
            for agent2 in agent_ids[i+1:]:
                traj1 = trajectories[agent1]
                traj2 = trajectories[agent2]
                
                # Count close encounters
                interactions = 0
                for t in range(min(len(traj1), len(traj2))):
                    pos1 = np.array(traj1[t])
                    pos2 = np.array(traj2[t])
                    
                    if np.linalg.norm(pos1 - pos2) < interaction_threshold:
                        interactions += 1
                
                if interactions >= min_interactions:
                    G.add_edge(agent1, agent2, weight=interactions)
        
        # Calculate network metrics
        metrics = {}
        
        if G.number_of_edges() > 0:
            metrics["density"] = nx.density(G)
            metrics["clustering_coefficient"] = nx.average_clustering(G)
            metrics["average_path_length"] = (
                nx.average_shortest_path_length(G) 
                if nx.is_connected(G) else float('inf')
            )
            metrics["degree_centrality"] = dict(nx.degree_centrality(G))
        else:
            metrics = {"density": 0, "clustering_coefficient": 0, "average_path_length": float('inf')}
        
        return metrics
    
    def _analyze_temporal_dynamics(self, trajectories: Dict[int, List[Tuple[float, float]]]) -> Dict[str, Any]:
        """Analyze temporal dynamics of the swarm."""
        if not trajectories:
            return {}
        
        # Calculate activity levels over time
        max_time = max(len(traj) for traj in trajectories.values())
        activity_levels = []
        
        for t in range(max_time):
            total_movement = 0
            active_agents = 0
            
            for agent_id, traj in trajectories.items():
                if t < len(traj) and t > 0:
                    pos_current = np.array(traj[t])
                    pos_previous = np.array(traj[t-1])
                    
                    movement = np.linalg.norm(pos_current - pos_previous)
                    total_movement += movement
                    
                    if movement > 0.1:  # Threshold for being considered "active"
                        active_agents += 1
            
            activity_levels.append({
                "timestep": t,
                "total_movement": total_movement,
                "active_agents": active_agents
            })
        
        return {
            "activity_levels": activity_levels,
            "peak_activity": max(activity_levels, key=lambda x: x["total_movement"]) if activity_levels else None,
            "average_activity": np.mean([a["total_movement"] for a in activity_levels]) if activity_levels else 0
        }
    
    def _analyze_spatial_distribution(self, trajectories: Dict[int, List[Tuple[float, float]]]) -> Dict[str, Any]:
        """Analyze spatial distribution patterns."""
        if not trajectories:
            return {}
        
        # Collect all positions
        all_positions = []
        for traj in trajectories.values():
            all_positions.extend(traj)
        
        if not all_positions:
            return {}
        
        positions_array = np.array(all_positions)
        
        # Calculate spatial statistics
        centroid = np.mean(positions_array, axis=0)
        spread = np.std(positions_array, axis=0)
        max_distance = np.max(np.linalg.norm(positions_array - centroid, axis=1))
        
        return {
            "centroid": centroid.tolist(),
            "spread": spread.tolist(),
            "max_distance_from_centroid": float(max_distance),
            "bounding_box": {
                "min_x": float(positions_array[:, 0].min()),
                "max_x": float(positions_array[:, 0].max()),
                "min_y": float(positions_array[:, 1].min()),
                "max_y": float(positions_array[:, 1].max())
            }
        }