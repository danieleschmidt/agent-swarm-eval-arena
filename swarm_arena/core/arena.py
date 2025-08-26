"""Main Arena class for orchestrating multi-agent simulations."""

from typing import Dict, List, Any, Optional, Type, Tuple
import numpy as np
import time
from dataclasses import dataclass, field

from .agent import BaseAgent, Agent, AgentState
from .environment import Environment, ForagingEnvironment
from .config import SwarmConfig
from ..utils.physics import SimplePhysicsEngine
from ..utils.seeding import set_global_seed
from ..utils.spatial import create_spatial_index
from ..utils.error_handler import safe_action_execution, with_error_handling, error_manager


@dataclass
class SimulationResults:
    """Results from a simulation run."""
    
    episode_rewards: Dict[int, List[float]] = field(default_factory=dict)
    agent_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    environment_stats: Dict[str, Any] = field(default_factory=dict)
    total_steps: int = 0
    episode_length: int = 0
    mean_reward: float = 0.0
    fairness_index: Optional[float] = None
    emergent_patterns: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Calculate derived statistics."""
        if self.episode_rewards:
            all_rewards = []
            for agent_rewards in self.episode_rewards.values():
                all_rewards.extend(agent_rewards)
            self.mean_reward = np.mean(all_rewards) if all_rewards else 0.0
            
            # Calculate basic fairness index (Gini coefficient approximation)
            if len(self.episode_rewards) > 1:
                agent_totals = [sum(rewards) for rewards in self.episode_rewards.values()]
                mean_total = np.mean(agent_totals)
                if mean_total > 0:
                    self.fairness_index = 1.0 - (np.std(agent_totals) / mean_total)


class Arena:
    """Main arena for multi-agent simulations.
    
    Orchestrates interactions between agents and environment, handles physics,
    collision detection, and maintains simulation state.
    """
    
    def __init__(self, config: SwarmConfig, environment: Optional[Environment] = None) -> None:
        """Initialize the arena.
        
        Args:
            config: Swarm configuration
            environment: Environment instance (defaults to ForagingEnvironment)
        """
        self.config = config
        self.environment = environment or ForagingEnvironment(config)
        
        # Set global seed for reproducibility
        if config.seed is not None:
            set_global_seed(config.seed)
        
        # Initialize physics engine
        self.physics = SimplePhysicsEngine(config)
        
        # Initialize spatial indexing for efficient neighbor queries
        spatial_type = "hash_grid" if config.num_agents < 1000 else "quadtree"
        self.spatial_index = create_spatial_index(config, spatial_type)
        
        # Agent management
        self.agents: Dict[int, BaseAgent] = {}
        self.agent_positions: Dict[int, np.ndarray] = {}
        self.agent_velocities: Dict[int, np.ndarray] = {}
        
        # Simulation state
        self.current_step = 0
        self.episode_rewards: Dict[int, List[float]] = {}
        
        # Performance tracking with bounded history
        self.step_times: List[float] = []
        self._max_performance_history = 1000  # Prevent memory growth
        
        # Communication tracking for research
        self.communication_signals: List[Any] = []
        self.record_communication = False
        
    def add_agents(self, agent_class: Type[BaseAgent], count: int, **agent_kwargs: Any) -> None:
        """Add multiple agents of the same type to the arena.
        
        Args:
            agent_class: Class of agents to create
            count: Number of agents to add
            **agent_kwargs: Additional arguments for agent initialization
        """
        start_id = len(self.agents)
        
        for i in range(count):
            agent_id = start_id + i
            
            # Generate random initial position
            initial_position = np.array([
                np.random.uniform(50, self.config.arena_size[0] - 50),
                np.random.uniform(50, self.config.arena_size[1] - 50)
            ])
            
            # Create agent
            agent = agent_class(agent_id, initial_position, **agent_kwargs)
            
            self.agents[agent_id] = agent
            self.agent_positions[agent_id] = initial_position.copy()
            self.agent_velocities[agent_id] = np.zeros(2, dtype=np.float32)
            self.episode_rewards[agent_id] = []
            
            # Add to spatial index
            self.spatial_index.update_agent(agent_id, initial_position)
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Add a single agent to the arena.
        
        Args:
            agent: Agent instance to add
        """
        agent_id = agent.agent_id
        self.agents[agent_id] = agent
        self.agent_positions[agent_id] = agent.state.position.copy()
        self.agent_velocities[agent_id] = agent.state.velocity.copy()
        self.episode_rewards[agent_id] = []
        
        # Add to spatial index
        self.spatial_index.update_agent(agent_id, agent.state.position)
    
    def reset(self) -> Dict[str, Any]:
        """Reset the arena to initial state.
        
        Returns:
            Initial state information
        """
        # Reset environment
        env_state = self.environment.reset()
        
        # Reset agents
        for agent_id, agent in self.agents.items():
            # Generate new random initial position
            initial_position = np.array([
                np.random.uniform(50, self.config.arena_size[0] - 50),
                np.random.uniform(50, self.config.arena_size[1] - 50)
            ])
            
            agent.reset(initial_position)
            self.agent_positions[agent_id] = initial_position.copy()
            self.agent_velocities[agent_id] = np.zeros(2, dtype=np.float32)
            self.episode_rewards[agent_id] = []
        
        # Reset simulation state
        self.current_step = 0
        self.step_times.clear()
        
        # Clear episode rewards history to prevent memory accumulation
        for agent_id in self.episode_rewards:
            self.episode_rewards[agent_id].clear()
        
        # Reset spatial index
        self.spatial_index.clear()
        
        # Re-populate spatial index with reset positions
        for agent_id, position in self.agent_positions.items():
            self.spatial_index.update_agent(agent_id, position)
        
        return {
            "num_agents": len(self.agents),
            "arena_size": self.config.arena_size,
            "environment": env_state
        }
    
    def step(self) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, float], bool, Dict[str, Any]]:
        """Execute one simulation step.
        
        Returns:
            Tuple of (observations, rewards, done, info)
        """
        step_start_time = time.time()
        
        # Get observations for all agents
        observations = self._get_observations()
        
        # Get actions from all agents with error handling
        actions = {}
        for agent_id, agent in self.agents.items():
            if agent.state.alive:
                action = safe_action_execution(agent, observations[agent_id])
                actions[agent_id] = action
        
        # Execute physics step
        self._execute_physics_step(actions)
        
        # Execute environment step
        env_obs, env_rewards, env_done, env_info = self.environment.step(actions)
        
        # Calculate total rewards
        rewards = self._calculate_rewards(actions, env_rewards)
        
        # Update agent records
        for agent_id, agent in self.agents.items():
            if agent_id in actions and agent_id in rewards:
                agent.record_action(actions[agent_id], rewards[agent_id])
                self.episode_rewards[agent_id].append(rewards[agent_id])
        
        # Check if simulation is done
        self.current_step += 1
        done = (self.current_step >= self.config.episode_length or 
                env_done or 
                not any(agent.state.alive for agent in self.agents.values()))
        
        # Performance tracking with memory bounds
        step_time = time.time() - step_start_time
        self.step_times.append(step_time)
        
        # Keep only recent performance history to prevent memory growth
        if len(self.step_times) > self._max_performance_history:
            self.step_times = self.step_times[-self._max_performance_history//2:]
        
        info = {
            "step": self.current_step,
            "active_agents": sum(1 for agent in self.agents.values() if agent.state.alive),
            "step_time": step_time,
            "average_step_time": np.mean(self.step_times[-100:]),  # Last 100 steps
            **env_info
        }
        
        return observations, rewards, done, info
    
    def run(self, episodes: int = 1, verbose: bool = True) -> SimulationResults:
        """Run the simulation for multiple episodes.
        
        Args:
            episodes: Number of episodes to run
            verbose: Whether to print progress information
            
        Returns:
            Simulation results
        """
        all_episode_rewards = {agent_id: [] for agent_id in self.agents.keys()}
        
        for episode in range(episodes):
            if verbose:
                print(f"Running episode {episode + 1}/{episodes}")
            
            # Reset for new episode
            self.reset()
            
            # Run episode
            done = False
            while not done:
                observations, rewards, done, info = self.step()
                
                if verbose and self.current_step % 100 == 0:
                    active_agents = info.get("active_agents", 0)
                    avg_reward = np.mean(list(rewards.values())) if rewards else 0.0
                    print(f"  Step {self.current_step}: {active_agents} active agents, "
                          f"avg reward: {avg_reward:.3f}")
            
            # Collect episode rewards
            for agent_id in self.agents.keys():
                episode_total = sum(self.episode_rewards[agent_id])
                all_episode_rewards[agent_id].append(episode_total)
        
        # Create results
        results = SimulationResults(
            episode_rewards=all_episode_rewards,
            agent_stats={agent_id: agent.get_stats() for agent_id, agent in self.agents.items()},
            environment_stats=self.environment.get_stats(),
            total_steps=self.current_step * episodes,
            episode_length=self.current_step,
        )
        
        if verbose:
            print(f"\nSimulation complete!")
            print(f"Mean reward per episode: {results.mean_reward:.3f}")
            if results.fairness_index is not None:
                print(f"Fairness index: {results.fairness_index:.3f}")
        
        return results
    
    def evaluate(self, num_episodes: int = 10, metrics: List[str] = None, 
                record_trajectories: bool = False) -> SimulationResults:
        """Evaluate the current agent configuration.
        
        Args:
            num_episodes: Number of episodes for evaluation
            metrics: List of metrics to compute
            record_trajectories: Whether to record agent trajectories
            
        Returns:
            Evaluation results
        """
        if metrics is None:
            metrics = ["efficiency", "fairness", "emergence"]
        
        results = self.run(episodes=num_episodes, verbose=False)
        
        # Add requested metrics
        if "efficiency" in metrics:
            # Calculate resource collection efficiency
            total_resources = results.environment_stats.get("collected_resources", 0)
            results.environment_stats["efficiency"] = total_resources / len(self.agents)
        
        if "emergence" in metrics:
            # Enhanced emergence detection with pattern analysis
            results.emergent_patterns = self._detect_emergent_patterns(results)
        
        if "coordination" in metrics:
            # Calculate coordination index based on agent clustering
            results.environment_stats["coordination"] = self._calculate_coordination_index()
        
        return results
    
    def _get_observations(self) -> Dict[int, Dict[str, Any]]:
        """Get observations for all agents.
        
        Returns:
            Dictionary mapping agent_id to observation
        """
        observations = {}
        
        for agent_id in self.agents.keys():
            if not self.agents[agent_id].state.alive:
                continue
                
            agent_pos = self.agent_positions[agent_id]
            
            # Get nearby agents using spatial index (O(1) average case)
            neighbor_ids = self.spatial_index.query_neighbors_fast(
                agent_id, self.config.observation_radius
            )
            
            # Convert to position list, filtering for alive agents
            nearby_agents = []
            for other_id in neighbor_ids:
                if (other_id in self.agents and 
                    self.agents[other_id].state.alive):
                    nearby_agents.append(self.agent_positions[other_id].tolist())
            
            # Get environment-specific observation
            env_obs = self.environment.get_observation_for_agent(agent_id, agent_pos)
            
            # Combine observations with standardized format
            observations[agent_id] = {
                **env_obs,
                "position": agent_pos.tolist(),  # Always include current position
                "velocity": self.agent_velocities[agent_id].tolist(),
                "nearby_agents": nearby_agents,
                "resources": env_obs.get("nearby_resources", []),
                "arena_bounds": {  # Standardized arena bounds format
                    "width": self.config.arena_size[0],
                    "height": self.config.arena_size[1]
                }
            }
        
        return observations
    
    def _execute_physics_step(self, actions: Dict[int, int]) -> None:
        """Execute physics simulation step.
        
        Args:
            actions: Dictionary of agent actions
        """
        # Convert actions to movement vectors
        action_to_movement = {
            0: np.array([0, 0]),      # no-op
            1: np.array([0, 1]),      # move_up
            2: np.array([0, -1]),     # move_down
            3: np.array([-1, 0]),     # move_left
            4: np.array([1, 0]),      # move_right
            5: np.array([0, 0]),      # collect_resource (no movement)
        }
        
        # Update velocities based on actions
        for agent_id, action in actions.items():
            if agent_id in self.agent_velocities:
                movement = action_to_movement.get(action, np.array([0, 0]))
                self.agent_velocities[agent_id] = movement * self.config.max_agent_speed
        
        # Update positions using simple integration
        dt = 1.0 / 60.0  # 60 FPS simulation
        
        for agent_id in self.agents.keys():
            if not self.agents[agent_id].state.alive:
                continue
                
            # Update position
            self.agent_positions[agent_id] += self.agent_velocities[agent_id] * dt
            
            # Enforce boundaries
            pos = self.agent_positions[agent_id]
            pos[0] = np.clip(pos[0], 0, self.config.arena_size[0])
            pos[1] = np.clip(pos[1], 0, self.config.arena_size[1])
            
            # Update agent state
            self.agents[agent_id].update_state(pos, self.agent_velocities[agent_id])
            
            # Update spatial index with new position
            self.spatial_index.update_agent(agent_id, pos)
        
        # Handle collisions if enabled
        if self.config.collision_detection:
            self._handle_collisions()
    
    def _handle_collisions(self) -> None:
        """Handle collisions between agents."""
        agent_ids = list(self.agents.keys())
        
        for i, agent_id_1 in enumerate(agent_ids):
            if not self.agents[agent_id_1].state.alive:
                continue
                
            for agent_id_2 in agent_ids[i+1:]:
                if not self.agents[agent_id_2].state.alive:
                    continue
                
                pos1 = self.agent_positions[agent_id_1]
                pos2 = self.agent_positions[agent_id_2]
                
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance < self.config.collision_radius:
                    # Simple elastic collision resolution
                    normal = (pos2 - pos1) / (distance + 1e-6)
                    
                    # Separate agents
                    overlap = self.config.collision_radius - distance
                    separation = normal * (overlap / 2)
                    
                    self.agent_positions[agent_id_1] -= separation
                    self.agent_positions[agent_id_2] += separation
                    
                    # Update velocities (simple bounce)
                    vel1 = self.agent_velocities[agent_id_1]
                    vel2 = self.agent_velocities[agent_id_2]
                    
                    self.agent_velocities[agent_id_1] = vel1 - 2 * np.dot(vel1, normal) * normal
                    self.agent_velocities[agent_id_2] = vel2 - 2 * np.dot(vel2, normal) * normal
    
    def _calculate_rewards(self, actions: Dict[int, int], env_rewards: Dict[int, float]) -> Dict[int, float]:
        """Calculate total rewards for all agents.
        
        Args:
            actions: Agent actions taken
            env_rewards: Rewards from environment
            
        Returns:
            Dictionary of total rewards per agent
        """
        rewards = env_rewards.copy()
        
        for agent_id, action in actions.items():
            if agent_id not in rewards:
                rewards[agent_id] = 0.0
            
            # Resource collection reward
            if action == 5:  # collect_resource
                collection_reward = self.environment.process_resource_collection(
                    agent_id, self.agent_positions[agent_id]
                )
                rewards[agent_id] += collection_reward * self.config.reward_config.get("resource_collection", 1.0)
                
                # Update agent's resource count
                if collection_reward > 0:
                    self.agents[agent_id].state.resources_collected += 1
            
            # Time penalty
            rewards[agent_id] += self.config.reward_config.get("time_penalty", -0.001)
            
            # Survival bonus
            if self.agents[agent_id].state.alive:
                rewards[agent_id] += self.config.reward_config.get("survival_bonus", 0.01)
        
        return rewards
    
    def run_episode(self, max_steps: int = 1000, record_communication: bool = False) -> Dict[str, Any]:
        """Run a single episode with detailed recording capabilities.
        
        Args:
            max_steps: Maximum steps in the episode
            record_communication: Whether to record communication signals
            
        Returns:
            Dictionary containing episode data
        """
        self.record_communication = record_communication
        self.communication_signals = []
        
        # Reset for new episode
        self.reset()
        
        episode_data = {
            "agent_positions": {aid: [] for aid in self.agents.keys()},
            "agent_actions": {aid: [] for aid in self.agents.keys()},
            "communication_signals": [],
            "rewards": {aid: [] for aid in self.agents.keys()},
            "steps": 0
        }
        
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Get observations
            observations = self._get_observations()
            
            # Collect agent actions
            actions = {}
            for agent_id, agent in self.agents.items():
                if agent.state.alive:
                    action = agent.act(observations[agent_id])
                    actions[agent_id] = action
                    
                    # Record action for research
                    if isinstance(action, dict):
                        # Handle communication-enabled actions
                        movement = action.get("movement", [0, 0])
                        communication = action.get("communication", [])
                        
                        episode_data["agent_actions"][agent_id].append(movement)
                        
                        # Record communication signals
                        if record_communication and communication:
                            episode_data["communication_signals"].extend(communication)
                    else:
                        episode_data["agent_actions"][agent_id].append(action)
            
            # Record positions
            for agent_id in self.agents.keys():
                if agent_id in self.agent_positions:
                    episode_data["agent_positions"][agent_id].append(
                        self.agent_positions[agent_id].copy()
                    )
            
            # Execute step
            obs, rewards, done, info = self.step()
            
            # Record rewards
            for agent_id, reward in rewards.items():
                episode_data["rewards"][agent_id].append(reward)
            
            step += 1
        
        episode_data["steps"] = step
        episode_data["steps"] = step
        return episode_data
    
    def _detect_emergent_patterns(self, results: SimulationResults) -> List[str]:
        """Detect emergent behavioral patterns from simulation results."""
        patterns = []
        
        try:
            # Analyze flocking behavior
            if len(self.agents) > 10:
                avg_resources_per_agent = sum(agent.state.resources_collected for agent in self.agents.values()) / len(self.agents)
                if avg_resources_per_agent > 5:
                    patterns.append("efficient_foraging")
                
                # Check for clustering
                positions = [agent.state.position for agent in self.agents.values() if agent.state.alive]
                if len(positions) > 5:
                    center = np.mean(positions, axis=0)
                    distances = [np.linalg.norm(pos - center) for pos in positions]
                    if np.std(distances) < 100:  # Agents are clustered
                        patterns.append("spatial_clustering")
        
        except Exception:
            patterns = ["basic_exploration"]
        
        return patterns if patterns else ["random_behavior"]
    
    def _calculate_coordination_index(self) -> float:
        """Calculate coordination index based on agent behavior similarity."""
        try:
            if len(self.agents) < 2:
                return 0.0
            
            # Analyze recent actions for coordination
            action_sequences = []
            for agent in self.agents.values():
                if len(agent.action_history) >= 10:
                    action_sequences.append(agent.action_history[-10:])
            
            if len(action_sequences) < 2:
                return 0.0
            
            # Calculate action similarity between agents
            similarities = []
            for i in range(len(action_sequences)):
                for j in range(i + 1, len(action_sequences)):
                    seq1, seq2 = action_sequences[i], action_sequences[j]
                    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
                    similarity = matches / len(seq1)
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
        
        except Exception:
            return 0.0