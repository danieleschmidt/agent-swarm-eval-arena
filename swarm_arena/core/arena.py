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
        
        # Get actions from all agents
        actions = {}
        for agent_id, agent in self.agents.items():
            if agent.state.alive:
                action = agent.act(observations[agent_id])
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
            metrics = ["efficiency", "fairness"]
        
        results = self.run(episodes=num_episodes, verbose=False)
        
        # Add requested metrics
        if "efficiency" in metrics:
            # Calculate resource collection efficiency
            total_resources = results.environment_stats.get("collected_resources", 0)
            results.environment_stats["efficiency"] = total_resources / len(self.agents)
        
        if "emergence" in metrics:
            # Placeholder for emergence detection
            results.emergent_patterns = ["flocking", "resource_clustering"]
        
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
            
            # Get nearby agents within observation radius
            nearby_agents = []
            for other_id, other_pos in self.agent_positions.items():
                if (other_id != agent_id and 
                    self.agents[other_id].state.alive and
                    np.linalg.norm(other_pos - agent_pos) <= self.config.observation_radius):
                    nearby_agents.append(other_pos.tolist())
            
            # Get environment-specific observation
            env_obs = self.environment.get_observation_for_agent(agent_id, agent_pos)
            
            # Combine observations
            observations[agent_id] = {
                **env_obs,
                "velocity": self.agent_velocities[agent_id].tolist(),
                "nearby_agents": nearby_agents,
                "resources": env_obs.get("nearby_resources", []),
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