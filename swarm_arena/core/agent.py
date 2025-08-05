"""Agent classes for the Swarm Arena."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from ..exceptions import AgentError, ValidationError
from ..utils.validation import validate_positive, validate_action, validate_position
from ..utils.logging import get_logger
import random
from collections import deque

logger = get_logger(__name__)


@dataclass
class AgentState:
    """State representation for an agent."""
    
    agent_id: int
    position: np.ndarray  # [x, y] coordinates
    velocity: np.ndarray  # [vx, vy] velocity vector
    health: float = 1.0
    energy: float = 1.0
    resources_collected: int = 0
    alive: bool = True
    last_action: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Ensure arrays are numpy arrays."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=np.float32)


class BaseAgent(ABC):
    """Abstract base class for all agents in the swarm arena."""
    
    def __init__(self, agent_id: int, initial_position: np.ndarray, **kwargs: Any) -> None:
        """Initialize the agent.
        
        Args:
            agent_id: Unique identifier for this agent
            initial_position: Starting position [x, y]
            **kwargs: Additional agent-specific parameters
            
        Raises:
            AgentError: If initialization parameters are invalid
        """
        try:
            # Validate inputs
            if not isinstance(agent_id, int) or agent_id < 0:
                raise AgentError(f"agent_id must be a non-negative integer, got {agent_id}")
            
            if not isinstance(initial_position, np.ndarray):
                try:
                    initial_position = np.array(initial_position, dtype=np.float32)
                except Exception:
                    raise AgentError(f"initial_position must be array-like, got {type(initial_position)}")
            
            if len(initial_position) != 2:
                raise AgentError(f"initial_position must be 2D, got {len(initial_position)}D")
            
            self.agent_id = agent_id
            self.state = AgentState(
                agent_id=agent_id,
                position=np.array(initial_position, dtype=np.float32),
                velocity=np.zeros(2, dtype=np.float32)
            )
            self.config = kwargs
            
            # Action history for analysis
            self.action_history: List[int] = []
            self.reward_history: List[float] = []
            
            logger.debug(f"Initialized agent {agent_id} at position {initial_position}")
            
        except Exception as e:
            if isinstance(e, AgentError):
                raise
            raise AgentError(f"Failed to initialize agent {agent_id}: {str(e)}")
        
    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> int:
        """Choose an action based on the current observation.
        
        Args:
            observation: Dictionary containing:
                - 'position': Agent's current position
                - 'velocity': Agent's current velocity  
                - 'nearby_agents': List of nearby agent positions
                - 'resources': List of nearby resource positions
                - 'arena_bounds': Arena boundary information
                - 'messages': Communication messages (if enabled)
        
        Returns:
            Action integer (0: no-op, 1: move_up, 2: move_down, 
                          3: move_left, 4: move_right, 5: collect_resource)
        """
        pass
    
    def reset(self, initial_position: np.ndarray) -> None:
        """Reset agent to initial state.
        
        Args:
            initial_position: New starting position
            
        Raises:
            AgentError: If reset parameters are invalid
        """
        try:
            if not isinstance(initial_position, np.ndarray):
                try:
                    initial_position = np.array(initial_position, dtype=np.float32)
                except Exception:
                    raise AgentError(f"initial_position must be array-like, got {type(initial_position)}")
            
            if len(initial_position) != 2:
                raise AgentError(f"initial_position must be 2D, got {len(initial_position)}D")
            
            self.state = AgentState(
                agent_id=self.agent_id,
                position=np.array(initial_position, dtype=np.float32),
                velocity=np.zeros(2, dtype=np.float32)
            )
            self.action_history.clear()
            self.reward_history.clear()
            
            logger.debug(f"Reset agent {self.agent_id} to position {initial_position}")
            
        except Exception as e:
            if isinstance(e, AgentError):
                raise
            raise AgentError(f"Failed to reset agent {self.agent_id}: {str(e)}")
    
    def update_state(self, new_position: np.ndarray, new_velocity: np.ndarray) -> None:
        """Update agent's physical state.
        
        Args:
            new_position: Updated position
            new_velocity: Updated velocity
            
        Raises:
            AgentError: If state update parameters are invalid
        """
        try:
            # Validate inputs
            if len(new_position) != 2:
                raise AgentError(f"new_position must be 2D, got {len(new_position)}D")
            
            if len(new_velocity) != 2:
                raise AgentError(f"new_velocity must be 2D, got {len(new_velocity)}D")
            
            # Check for NaN or infinite values
            if not np.all(np.isfinite(new_position)):
                raise AgentError(f"new_position contains invalid values: {new_position}")
            
            if not np.all(np.isfinite(new_velocity)):
                raise AgentError(f"new_velocity contains invalid values: {new_velocity}")
            
            self.state.position = np.array(new_position, dtype=np.float32)
            self.state.velocity = np.array(new_velocity, dtype=np.float32)
            
        except Exception as e:
            if isinstance(e, AgentError):
                raise
            raise AgentError(f"Failed to update state for agent {self.agent_id}: {str(e)}")
    
    def record_action(self, action: int, reward: float) -> None:
        """Record action and reward for analysis.
        
        Args:
            action: Action taken
            reward: Reward received
            
        Raises:
            AgentError: If action or reward are invalid
        """
        try:
            # Validate action
            validate_action(action)
            
            # Validate reward
            if not isinstance(reward, (int, float)) or not np.isfinite(reward):
                raise AgentError(f"reward must be a finite number, got {reward}")
            
            self.action_history.append(action)
            self.reward_history.append(float(reward))
            self.state.last_action = action
            
        except Exception as e:
            if isinstance(e, (AgentError, ValidationError)):
                raise AgentError(f"Failed to record action for agent {self.agent_id}: {str(e)}")
            raise AgentError(f"Unexpected error recording action for agent {self.agent_id}: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics for analysis.
        
        Returns:
            Dictionary of agent statistics
        """
        return {
            "agent_id": self.agent_id,
            "total_reward": sum(self.reward_history),
            "average_reward": np.mean(self.reward_history) if self.reward_history else 0.0,
            "resources_collected": self.state.resources_collected,
            "final_position": self.state.position.tolist(),
            "alive": self.state.alive,
            "health": self.state.health,
            "energy": self.state.energy,
            "actions_taken": len(self.action_history),
        }


class Agent(BaseAgent):
    """Default implementation of BaseAgent with basic behaviors."""
    
    def __init__(self, agent_id: int, initial_position: np.ndarray, **kwargs: Any) -> None:
        super().__init__(agent_id, initial_position, **kwargs)
        self.exploration_rate = kwargs.get("exploration_rate", 0.1)
        self.cooperation_tendency = kwargs.get("cooperation_tendency", 0.5)
        
    def act(self, observation: Dict[str, Any]) -> int:
        """Simple rule-based action selection.
        
        Priority:
        1. Collect nearby resources
        2. Move towards resources if visible
        3. Move towards other agents if cooperative
        4. Random exploration
        
        Raises:
            AgentError: If observation is invalid or action selection fails
        """
        try:
            # Validate observation
            if not isinstance(observation, dict):
                raise AgentError(f"observation must be a dictionary, got {type(observation)}")
            
            # Extract observation components with validation
            position = observation.get("position", self.state.position.tolist())
            if not isinstance(position, (list, np.ndarray)) or len(position) != 2:
                raise AgentError(f"Invalid position in observation: {position}")
            
            position = np.array(position)
            resources = observation.get("resources", [])
            nearby_agents = observation.get("nearby_agents", [])
            arena_bounds = observation.get("arena_bounds", {"width": 1000, "height": 1000})
            
            # Check if there's a resource at current position (collect action)
            for resource_pos in resources:
                if np.linalg.norm(np.array(resource_pos) - position) < 10.0:
                    return 5  # collect_resource
            
            # Move towards nearest resource
            if resources:
                nearest_resource = min(resources, key=lambda r: np.linalg.norm(np.array(r) - position))
                return self._move_towards(position, nearest_resource)
            
            # Cooperative behavior: move towards center of nearby agents
            if nearby_agents and np.random.random() < self.cooperation_tendency:
                center = np.mean(nearby_agents, axis=0)
                return self._move_towards(position, center)
            
            # Random exploration with boundary avoidance
            if np.random.random() < self.exploration_rate:
                # Avoid boundaries
                if position[0] < 50:  # too close to left
                    return 4  # move_right
                elif position[0] > arena_bounds["width"] - 50:  # too close to right
                    return 3  # move_left
                elif position[1] < 50:  # too close to bottom
                    return 1  # move_up
                elif position[1] > arena_bounds["height"] - 50:  # too close to top
                    return 2  # move_down
                else:
                    return np.random.randint(1, 5)  # random movement
            
            return 0  # no-op
            
        except Exception as e:
            if isinstance(e, AgentError):
                raise
            logger.error(f"Agent {self.agent_id} action selection failed", error=str(e))
            return 0  # Safe fallback: no-op
    
    def _move_towards(self, current_pos: np.ndarray, target_pos: np.ndarray) -> int:
        """Determine best action to move towards target.
        
        Args:
            current_pos: Current position
            target_pos: Target position
            
        Returns:
            Action to move towards target
            
        Raises:
            AgentError: If positions are invalid
        """
        try:
            current_pos = np.array(current_pos)
            target_pos = np.array(target_pos)
            
            if len(current_pos) != 2 or len(target_pos) != 2:
                raise AgentError("Positions must be 2D")
            
            if not np.all(np.isfinite(current_pos)) or not np.all(np.isfinite(target_pos)):
                raise AgentError("Positions must be finite")
            
            diff = target_pos - current_pos
            
            # Avoid division by zero or very small differences
            if np.linalg.norm(diff) < 1e-6:
                return 0  # no-op if already at target
            
            # Choose action based on largest difference
            if abs(diff[0]) > abs(diff[1]):
                return 4 if diff[0] > 0 else 3  # move_right or move_left
            else:
                return 1 if diff[1] > 0 else 2  # move_up or move_down
                
        except Exception as e:
            if isinstance(e, AgentError):
                raise
            raise AgentError(f"Move calculation failed: {str(e)}")


class CooperativeAgent(Agent):
    """Agent that prioritizes cooperation and group behavior."""
    
    def __init__(self, agent_id: int, initial_position: np.ndarray, **kwargs: Any) -> None:
        super().__init__(agent_id, initial_position, **kwargs)
        self.cooperation_tendency = 0.8  # High cooperation tendency
        
    def act(self, observation: Dict[str, Any]) -> int:
        """Cooperative action selection with group coordination.
        
        Raises:
            AgentError: If cooperative action selection fails
        """
        try:
            nearby_agents = observation.get("nearby_agents", [])
            
            # Always move towards group center if agents nearby
            if len(nearby_agents) > 0:
                # Validate nearby agents data
                for i, agent_pos in enumerate(nearby_agents):
                    if not isinstance(agent_pos, (list, np.ndarray)) or len(agent_pos) != 2:
                        raise AgentError(f"Invalid nearby agent position {i}: {agent_pos}")
                
                center = np.mean(nearby_agents, axis=0)
                return self._move_towards(observation["position"], center)
            
            # Fall back to parent behavior
            return super().act(observation)
            
        except Exception as e:
            if isinstance(e, AgentError):
                raise
            logger.error(f"Cooperative agent {self.agent_id} action failed", error=str(e))
            return 0  # Safe fallback


class CompetitiveAgent(Agent):
    """Agent that prioritizes individual resource collection."""
    
    def __init__(self, agent_id: int, initial_position: np.ndarray, **kwargs: Any) -> None:
        super().__init__(agent_id, initial_position, **kwargs)
        self.cooperation_tendency = 0.1  # Low cooperation tendency
        
    def act(self, observation: Dict[str, Any]) -> int:
        """Competitive action selection focused on resource acquisition.
        
        Raises:
            AgentError: If competitive action selection fails
        """
        try:
            resources = observation.get("resources", [])
            
            # Aggressively pursue resources
            if resources:
                # Validate resource positions
                valid_resources = []
                for i, resource_pos in enumerate(resources):
                    if isinstance(resource_pos, (list, np.ndarray)) and len(resource_pos) == 2:
                        if np.all(np.isfinite(resource_pos)):
                            valid_resources.append(resource_pos)
                        else:
                            logger.warning(f"Invalid resource position {i}: {resource_pos}")
                
                if valid_resources:
                    position = np.array(observation["position"])
                    nearest_resource = min(
                        valid_resources, 
                        key=lambda r: np.linalg.norm(np.array(r) - position)
                    )
                    return self._move_towards(position, nearest_resource)
            
            # Explore to find resources
            return np.random.randint(1, 5)
            
        except Exception as e:
            if isinstance(e, AgentError):
                raise
            logger.error(f"Competitive agent {self.agent_id} action failed", error=str(e))
            return 0  # Safe fallback


class RandomAgent(Agent):
    """Agent that takes random actions for baseline comparison."""
    
    def act(self, observation: Dict[str, Any]) -> int:
        """Random action selection.
        
        Raises:
            AgentError: If random action selection fails
        """
        try:
            return int(np.random.randint(0, 6))  # 0-5 for all possible actions
        except Exception as e:
            logger.error(f"Random agent {self.agent_id} action failed", error=str(e))
            return 0  # Safe fallback


class LearningAgent(Agent):
    """Agent that learns from experience using simple Q-learning."""
    
    def __init__(self, agent_id: int, initial_position: np.ndarray, 
                 learning_rate: float = 0.1, epsilon: float = 0.1, **kwargs: Any) -> None:
        super().__init__(agent_id, initial_position, **kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        self.q_table: Dict[str, np.ndarray] = {}  # State -> action values
        self.last_state: Optional[str] = None
        self.last_action: Optional[int] = None
        self.memory = deque(maxlen=1000)  # Experience replay buffer
        
    def _get_state_key(self, observation: Dict[str, Any]) -> str:
        """Convert observation to discrete state key."""
        try:
            position = np.array(observation.get("position", [0, 0]))
            nearby_agents = len(observation.get("nearby_agents", []))
            resources = len(observation.get("resources", []))
            
            # Discretize position to reduce state space
            pos_discrete = (int(position[0] // 50), int(position[1] // 50))
            
            # Create state key
            state_key = f"{pos_discrete}_{nearby_agents}_{resources}"
            return state_key
            
        except Exception as e:
            logger.warning(f"State key generation failed: {str(e)}")
            return "default_state"
    
    def _get_q_values(self, state: str) -> np.ndarray:
        """Get Q-values for state, initializing if needed."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(6, dtype=np.float32)  # 6 possible actions
        return self.q_table[state]
    
    def act(self, observation: Dict[str, Any]) -> int:
        """Action selection using epsilon-greedy Q-learning."""
        try:
            current_state = self._get_state_key(observation)
            q_values = self._get_q_values(current_state)
            
            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                action = np.random.randint(0, 6)  # Explore
            else:
                action = int(np.argmax(q_values))  # Exploit
            
            # Store for learning
            self.last_state = current_state
            self.last_action = action
            
            return action
            
        except Exception as e:
            logger.error(f"Learning agent {self.agent_id} action failed", error=str(e))
            return 0
    
    def learn(self, reward: float, next_observation: Dict[str, Any], done: bool = False) -> None:
        """Update Q-values based on experience."""
        try:
            if self.last_state is None or self.last_action is None:
                return
            
            next_state = self._get_state_key(next_observation)
            next_q_values = self._get_q_values(next_state)
            
            # Q-learning update
            old_q = self.q_table[self.last_state][self.last_action]
            next_max_q = 0.0 if done else np.max(next_q_values)
            
            new_q = old_q + self.learning_rate * (reward + 0.95 * next_max_q - old_q)
            self.q_table[self.last_state][self.last_action] = new_q
            
            # Store experience for potential replay
            self.memory.append({
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
        except Exception as e:
            logger.error(f"Learning update failed: {str(e)}")


class HierarchicalAgent(Agent):
    """Agent with hierarchical behavior planning."""
    
    def __init__(self, agent_id: int, initial_position: np.ndarray, **kwargs: Any) -> None:
        super().__init__(agent_id, initial_position, **kwargs)
        self.strategy = "explore"  # High-level strategy
        self.strategy_timer = 0
        self.strategy_duration = 100  # Steps per strategy
        self.target_position: Optional[np.ndarray] = None
        
    def _select_strategy(self, observation: Dict[str, Any]) -> str:
        """Select high-level strategy based on observation."""
        try:
            resources = observation.get("resources", [])
            nearby_agents = observation.get("nearby_agents", [])
            
            # Strategy selection logic
            if len(resources) > 3:
                return "harvest"  # Many resources available
            elif len(nearby_agents) > 5:
                return "cooperate"  # Many agents nearby
            elif len(resources) == 0:
                return "explore"  # No resources visible
            else:
                return "compete"  # Few resources, compete for them
                
        except Exception as e:
            logger.warning(f"Strategy selection failed: {str(e)}")
            return "explore"
    
    def _execute_strategy(self, observation: Dict[str, Any]) -> int:
        """Execute current strategy."""
        try:
            position = np.array(observation.get("position", [0, 0]))
            
            if self.strategy == "harvest":
                # Focus on resource collection
                resources = observation.get("resources", [])
                if resources:
                    nearest = min(resources, key=lambda r: np.linalg.norm(np.array(r) - position))
                    if np.linalg.norm(np.array(nearest) - position) < 10:
                        return 5  # collect_resource
                    return self._move_towards(position, nearest)
                return np.random.randint(1, 5)
                
            elif self.strategy == "cooperate":
                # Move towards group center
                nearby_agents = observation.get("nearby_agents", [])
                if nearby_agents:
                    center = np.mean(nearby_agents, axis=0)
                    return self._move_towards(position, center)
                return 0
                
            elif self.strategy == "explore":
                # Explore systematically
                if self.target_position is None or np.linalg.norm(position - self.target_position) < 20:
                    # Set new exploration target
                    arena_bounds = observation.get("arena_bounds", {"width": 800, "height": 600})
                    self.target_position = np.array([
                        np.random.uniform(50, arena_bounds["width"] - 50),
                        np.random.uniform(50, arena_bounds["height"] - 50)
                    ])
                return self._move_towards(position, self.target_position)
                
            elif self.strategy == "compete":
                # Aggressive resource competition
                resources = observation.get("resources", [])
                if resources:
                    nearest = min(resources, key=lambda r: np.linalg.norm(np.array(r) - position))
                    return self._move_towards(position, nearest)
                return np.random.randint(1, 5)
                
            return 0
            
        except Exception as e:
            logger.error(f"Strategy execution failed: {str(e)}")
            return 0
    
    def act(self, observation: Dict[str, Any]) -> int:
        """Hierarchical action selection."""
        try:
            # Update strategy periodically
            self.strategy_timer += 1
            if self.strategy_timer >= self.strategy_duration:
                self.strategy = self._select_strategy(observation)
                self.strategy_timer = 0
                self.target_position = None  # Reset target
            
            return self._execute_strategy(observation)
            
        except Exception as e:
            logger.error(f"Hierarchical agent {self.agent_id} action failed", error=str(e))
            return 0


class SwarmAgent(Agent):
    """Agent optimized for swarm intelligence behaviors."""
    
    def __init__(self, agent_id: int, initial_position: np.ndarray, **kwargs: Any) -> None:
        super().__init__(agent_id, initial_position, **kwargs)
        self.pheromone_trails: Dict[str, float] = {}  # Simple pheromone memory
        self.communication_range = 100.0
        self.follow_strength = 0.7
        self.leadership_tendency = np.random.random()  # Random leadership trait
        
    def _detect_swarm_patterns(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Detect emergent swarm patterns."""
        nearby_agents = observation.get("nearby_agents", [])
        if len(nearby_agents) < 3:
            return {"pattern": "isolated", "strength": 0.0}
        
        position = np.array(observation.get("position", [0, 0]))
        
        # Calculate swarm metrics
        agents_array = np.array(nearby_agents)
        center = np.mean(agents_array, axis=0)
        distances = [np.linalg.norm(pos - center) for pos in agents_array]
        avg_distance = np.mean(distances)
        
        # Detect flocking
        if avg_distance < 30 and len(nearby_agents) > 5:
            return {"pattern": "flocking", "strength": 0.8, "center": center}
        
        # Detect spreading
        if avg_distance > 80:
            return {"pattern": "spreading", "strength": 0.6}
        
        return {"pattern": "neutral", "strength": 0.3}
    
    def act(self, observation: Dict[str, Any]) -> int:
        """Swarm-intelligent action selection."""
        try:
            pattern_info = self._detect_swarm_patterns(observation)
            position = np.array(observation.get("position", [0, 0]))
            
            # Swarm behavior based on detected patterns
            if pattern_info["pattern"] == "flocking":
                # Maintain cohesion while avoiding overcrowding
                center = pattern_info.get("center", position)
                distance_to_center = np.linalg.norm(position - center)
                
                if distance_to_center > 50:  # Too far from swarm
                    return self._move_towards(position, center)
                elif distance_to_center < 15:  # Too close, spread out
                    # Move away from center
                    diff = position - center
                    if np.linalg.norm(diff) > 0:
                        away_point = position + diff / np.linalg.norm(diff) * 20
                        return self._move_towards(position, away_point)
                    return np.random.randint(1, 5)
                
            elif pattern_info["pattern"] == "spreading":
                # Try to reform swarm
                nearby_agents = observation.get("nearby_agents", [])
                if nearby_agents:
                    closest = min(nearby_agents, key=lambda a: np.linalg.norm(np.array(a) - position))
                    return self._move_towards(position, closest)
            
            # Default swarm behavior: balance exploration and resource collection
            resources = observation.get("resources", [])
            if resources and np.random.random() < 0.6:
                nearest_resource = min(resources, key=lambda r: np.linalg.norm(np.array(r) - position))
                if np.linalg.norm(np.array(nearest_resource) - position) < 10:
                    return 5  # collect_resource
                return self._move_towards(position, nearest_resource)
            
            # Exploration with swarm influence
            if self.leadership_tendency > 0.7:  # Leader behavior
                return np.random.randint(1, 5)  # Explore new areas
            else:  # Follower behavior
                nearby_agents = observation.get("nearby_agents", [])
                if nearby_agents:
                    # Follow the most central agent
                    center = np.mean(nearby_agents, axis=0)
                    return self._move_towards(position, center)
                return np.random.randint(1, 5)
                
        except Exception as e:
            logger.error(f"Swarm agent {self.agent_id} action failed", error=str(e))
            return 0


class AdaptiveAgent(Agent):
    """Agent that adapts its behavior based on environmental feedback."""
    
    def __init__(self, agent_id: int, initial_position: np.ndarray, **kwargs: Any) -> None:
        super().__init__(agent_id, initial_position, **kwargs)
        self.behavior_weights = {
            "explore": 0.25,
            "collect": 0.25, 
            "cooperate": 0.25,
            "compete": 0.25
        }
        self.performance_history = deque(maxlen=50)
        self.adaptation_rate = 0.05
        self.last_reward = 0.0
        
    def update_behavior_weights(self, reward: float) -> None:
        """Adapt behavior weights based on reward feedback."""
        try:
            reward_delta = reward - self.last_reward
            self.performance_history.append(reward_delta)
            
            # Simple adaptation: increase weight of successful behaviors
            if len(self.performance_history) >= 10:
                recent_performance = np.mean(list(self.performance_history)[-10:])
                
                if recent_performance > 0.01:  # Good performance
                    # Reinforce current dominant behavior
                    max_behavior = max(self.behavior_weights, key=self.behavior_weights.get)
                    self.behavior_weights[max_behavior] = min(0.7, 
                        self.behavior_weights[max_behavior] + self.adaptation_rate)
                elif recent_performance < -0.01:  # Poor performance
                    # Diversify behaviors
                    for behavior in self.behavior_weights:
                        self.behavior_weights[behavior] = 0.25
                
                # Normalize weights
                total = sum(self.behavior_weights.values())
                self.behavior_weights = {k: v/total for k, v in self.behavior_weights.items()}
            
            self.last_reward = reward
            
        except Exception as e:
            logger.warning(f"Behavior adaptation failed: {str(e)}")
    
    def act(self, observation: Dict[str, Any]) -> int:
        """Adaptive action selection based on learned behavior weights."""
        try:
            # Select behavior based on weights
            behaviors = list(self.behavior_weights.keys())
            weights = list(self.behavior_weights.values())
            selected_behavior = np.random.choice(behaviors, p=weights)
            
            position = np.array(observation.get("position", [0, 0]))
            resources = observation.get("resources", [])
            nearby_agents = observation.get("nearby_agents", [])
            
            if selected_behavior == "explore":
                # Exploration behavior
                arena_bounds = observation.get("arena_bounds", {"width": 800, "height": 600})
                if (position[0] < 100 or position[0] > arena_bounds["width"] - 100 or
                    position[1] < 100 or position[1] > arena_bounds["height"] - 100):
                    # Move towards center if near boundaries
                    center = np.array([arena_bounds["width"]/2, arena_bounds["height"]/2])
                    return self._move_towards(position, center)
                return np.random.randint(1, 5)
                
            elif selected_behavior == "collect":
                # Resource collection behavior
                if resources:
                    nearest = min(resources, key=lambda r: np.linalg.norm(np.array(r) - position))
                    if np.linalg.norm(np.array(nearest) - position) < 10:
                        return 5  # collect_resource
                    return self._move_towards(position, nearest)
                return np.random.randint(1, 5)
                
            elif selected_behavior == "cooperate":
                # Cooperative behavior
                if nearby_agents:
                    center = np.mean(nearby_agents, axis=0)
                    return self._move_towards(position, center)
                return 0
                
            elif selected_behavior == "compete":
                # Competitive behavior
                if resources and nearby_agents:
                    # Rush to nearest resource
                    nearest = min(resources, key=lambda r: np.linalg.norm(np.array(r) - position))
                    return self._move_towards(position, nearest)
                return np.random.randint(1, 5)
            
            return 0
            
        except Exception as e:
            logger.error(f"Adaptive agent {self.agent_id} action failed", error=str(e))
            return 0