"""Agent classes for the Swarm Arena."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from ..exceptions import AgentError, ValidationError
from ..utils.validation import validate_positive, validate_action, validate_position
from ..utils.logging import get_logger

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