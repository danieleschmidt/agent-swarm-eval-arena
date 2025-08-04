"""Environment implementations for the Swarm Arena."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
from .config import SwarmConfig
from ..exceptions import EnvironmentError, ResourceError
from ..utils.validation import validate_positive, validate_position
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Resource:
    """Resource object in the environment."""
    
    position: np.ndarray
    value: float = 1.0
    resource_type: str = "generic"
    respawn_time: int = 100
    collected: bool = False
    
    def __post_init__(self) -> None:
        """Ensure position is numpy array."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)


@dataclass 
class EnvironmentState:
    """Complete state of the environment."""
    
    step: int = 0
    resources: List[Resource] = field(default_factory=list)
    boundaries: Dict[str, float] = field(default_factory=lambda: {
        "min_x": 0, "max_x": 1000, "min_y": 0, "max_y": 1000
    })
    active_effects: List[Dict[str, Any]] = field(default_factory=list)


class Environment(ABC):
    """Abstract base class for all environments in the swarm arena."""
    
    def __init__(self, config: SwarmConfig) -> None:
        """Initialize the environment.
        
        Args:
            config: Swarm configuration object
            
        Raises:
            EnvironmentError: If initialization fails
        """
        try:
            if not isinstance(config, SwarmConfig):
                raise EnvironmentError(f"config must be SwarmConfig instance, got {type(config)}")
            
            self.config = config
            self.state = EnvironmentState()
            self.state.boundaries = {
                "min_x": 0,
                "max_x": config.arena_size[0],
                "min_y": 0, 
                "max_y": config.arena_size[1]
            }
            
            # Set random seed if provided
            if config.seed is not None:
                np.random.seed(config.seed)
                logger.debug(f"Environment initialized with seed {config.seed}")
            
            logger.info(f"Environment initialized with arena size {config.arena_size}")
            
        except Exception as e:
            if isinstance(e, EnvironmentError):
                raise
            raise EnvironmentError(f"Failed to initialize environment: {str(e)}")
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state.
        
        Returns:
            Initial environment state
        """
        pass
    
    @abstractmethod
    def step(self, actions: Dict[int, int]) -> Tuple[Dict[str, Any], Dict[int, float], bool, Dict[str, Any]]:
        """Execute one environment step.
        
        Args:
            actions: Dictionary mapping agent_id -> action
            
        Returns:
            Tuple of (observations, rewards, done, info)
        """
        pass
    
    def get_observation_for_agent(self, agent_id: int, agent_position: np.ndarray) -> Dict[str, Any]:
        """Get observation for a specific agent.
        
        Args:
            agent_id: ID of the agent
            agent_position: Current position of the agent
            
        Returns:
            Observation dictionary for this agent
            
        Raises:
            EnvironmentError: If observation generation fails
        """
        try:
            # Validate inputs
            if not isinstance(agent_id, int) or agent_id < 0:
                raise EnvironmentError(f"agent_id must be non-negative integer, got {agent_id}")
            
            if not isinstance(agent_position, np.ndarray) or len(agent_position) != 2:
                raise EnvironmentError(f"agent_position must be 2D array, got {agent_position}")
            
            if not np.all(np.isfinite(agent_position)):
                raise EnvironmentError(f"agent_position contains invalid values: {agent_position}")
            
            # Get nearby resources within observation radius
            nearby_resources = []
            for resource in self.state.resources:
                if not resource.collected:
                    try:
                        distance = np.linalg.norm(resource.position - agent_position)
                        if distance <= self.config.observation_radius:
                            nearby_resources.append(resource.position.tolist())
                    except Exception as e:
                        logger.warning(f"Error calculating distance to resource", error=str(e))
                        continue
            
            return {
                "position": agent_position.tolist(),
                "nearby_resources": nearby_resources,
                "arena_bounds": self.state.boundaries,
                "step": self.state.step,
            }
            
        except Exception as e:
            if isinstance(e, EnvironmentError):
                raise
            raise EnvironmentError(f"Failed to get observation for agent {agent_id}: {str(e)}")
    
    def spawn_resources(self) -> None:
        """Spawn new resources based on spawn rate.
        
        Raises:
            ResourceError: If resource spawning fails
        """
        try:
            if np.random.random() < self.config.resource_spawn_rate:
                # Generate valid position within boundaries
                min_x, max_x = self.state.boundaries["min_x"], self.state.boundaries["max_x"]
                min_y, max_y = self.state.boundaries["min_y"], self.state.boundaries["max_y"]
                
                # Add small buffer from edges
                buffer = 10.0
                safe_min_x = min_x + buffer
                safe_max_x = max_x - buffer
                safe_min_y = min_y + buffer
                safe_max_y = max_y - buffer
                
                if safe_max_x <= safe_min_x or safe_max_y <= safe_min_y:
                    logger.warning("Arena too small for resource spawning with buffer")
                    return
                
                position = np.array([
                    np.random.uniform(safe_min_x, safe_max_x),
                    np.random.uniform(safe_min_y, safe_max_y)
                ])
                
                # Validate position
                validate_position(position, self.config.arena_size, "resource_position")
                
                new_resource = Resource(
                    position=position,
                    value=max(0.1, np.random.uniform(0.5, 2.0)),  # Ensure positive value
                    resource_type="energy"
                )
                
                self.state.resources.append(new_resource)
                logger.debug(f"Spawned resource at {position} with value {new_resource.value}")
                
        except Exception as e:
            if isinstance(e, ResourceError):
                raise
            raise ResourceError(f"Failed to spawn resource: {str(e)}")
    
    def collect_resource_at(self, position: np.ndarray, collection_radius: float = 10.0) -> Optional[Resource]:
        """Attempt to collect resource at given position.
        
        Args:
            position: Position to collect at
            collection_radius: Radius for collection detection
            
        Returns:
            Collected resource if successful, None otherwise
            
        Raises:
            ResourceError: If collection attempt fails
        """
        try:
            # Validate inputs
            if not isinstance(position, np.ndarray) or len(position) != 2:
                raise ResourceError(f"position must be 2D array, got {position}")
            
            if not np.all(np.isfinite(position)):
                raise ResourceError(f"position contains invalid values: {position}")
            
            validate_positive(collection_radius, "collection_radius")
            
            # Find collectible resource
            for resource in self.state.resources:
                if not resource.collected:
                    try:
                        distance = np.linalg.norm(resource.position - position)
                        if distance <= collection_radius:
                            resource.collected = True
                            logger.debug(f"Collected resource at {resource.position} with value {resource.value}")
                            return resource
                    except Exception as e:
                        logger.warning(f"Error checking resource collection", error=str(e))
                        continue
            
            return None
            
        except Exception as e:
            if isinstance(e, ResourceError):
                raise
            raise ResourceError(f"Failed to collect resource: {str(e)}")
    
    def is_position_valid(self, position: np.ndarray) -> bool:
        """Check if position is within environment boundaries.
        
        Args:
            position: Position to check
            
        Returns:
            True if position is valid, False otherwise
        """
        return (
            self.state.boundaries["min_x"] <= position[0] <= self.state.boundaries["max_x"] and
            self.state.boundaries["min_y"] <= position[1] <= self.state.boundaries["max_y"]
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics.
        
        Returns:
            Dictionary of environment stats
        """
        active_resources = sum(1 for r in self.state.resources if not r.collected)
        collected_resources = sum(1 for r in self.state.resources if r.collected)
        
        return {
            "step": self.state.step,
            "active_resources": active_resources,
            "collected_resources": collected_resources,
            "total_resources_spawned": len(self.state.resources),
            "resource_spawn_rate": self.config.resource_spawn_rate,
        }


class ForagingEnvironment(Environment):
    """Simple foraging environment where agents collect resources."""
    
    def __init__(self, config: SwarmConfig) -> None:
        super().__init__(config)
        self.collection_radius = 10.0
        
    def reset(self) -> Dict[str, Any]:
        """Reset foraging environment.
        
        Raises:
            EnvironmentError: If reset fails
        """
        try:
            self.state = EnvironmentState()
            self.state.boundaries = {
                "min_x": 0,
                "max_x": self.config.arena_size[0],
                "min_y": 0,
                "max_y": self.config.arena_size[1]
            }
            
            # Spawn initial resources with error handling
            initial_resources = max(10, self.config.num_agents // 5)
            successful_spawns = 0
            
            for attempt in range(initial_resources * 2):  # Allow more attempts
                try:
                    self.spawn_resources()
                    successful_spawns += 1
                    if successful_spawns >= initial_resources:
                        break
                except Exception as e:
                    logger.warning(f"Resource spawn attempt {attempt} failed", error=str(e))
                    continue
            
            if successful_spawns == 0:
                logger.warning("No initial resources spawned successfully")
            else:
                logger.info(f"Spawned {successful_spawns} initial resources")
            
            return {"environment": "foraging", "bounds": self.state.boundaries}
            
        except Exception as e:
            raise EnvironmentError(f"Failed to reset foraging environment: {str(e)}")
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict[str, Any], Dict[int, float], bool, Dict[str, Any]]:
        """Execute foraging environment step.
        
        Args:
            actions: Dictionary mapping agent_id -> action
            
        Returns:
            Environment step results
            
        Raises:
            EnvironmentError: If step execution fails
        """
        try:
            # Validate inputs
            if not isinstance(actions, dict):
                raise EnvironmentError(f"actions must be a dictionary, got {type(actions)}")
            
            for agent_id, action in actions.items():
                if not isinstance(agent_id, int) or agent_id < 0:
                    raise EnvironmentError(f"Invalid agent_id: {agent_id}")
                
                if not isinstance(action, int) or action < 0 or action > 5:
                    raise EnvironmentError(f"Invalid action {action} for agent {agent_id}")
            
            self.state.step += 1
            rewards = {agent_id: 0.0 for agent_id in actions.keys()}
            
            # Spawn new resources with error handling
            try:
                self.spawn_resources()
            except Exception as e:
                logger.warning(f"Resource spawning failed at step {self.state.step}", error=str(e))
            
            # Process collection actions (action 5)
            collections = {}
            for agent_id, action in actions.items():
                collections[agent_id] = (action == 5)
            
            # Check if episode is done
            done = self.state.step >= self.config.episode_length
            
            info = {
                "collections": collections,
                "resources_available": sum(1 for r in self.state.resources if not r.collected),
                "environment_stats": self.get_stats()
            }
            
            return {}, rewards, done, info
            
        except Exception as e:
            if isinstance(e, EnvironmentError):
                raise
            raise EnvironmentError(f"Failed to execute environment step: {str(e)}")
    
    def process_resource_collection(self, agent_id: int, agent_position: np.ndarray) -> float:
        """Process resource collection for an agent.
        
        Args:
            agent_id: ID of collecting agent
            agent_position: Position of the agent
            
        Returns:
            Reward from collection
            
        Raises:
            EnvironmentError: If collection processing fails
        """
        try:
            collected_resource = self.collect_resource_at(agent_position, self.collection_radius)
            if collected_resource:
                logger.debug(f"Agent {agent_id} collected resource worth {collected_resource.value}")
                return collected_resource.value
            return 0.0
            
        except Exception as e:
            logger.error(f"Resource collection failed for agent {agent_id}", error=str(e))
            return 0.0  # Return 0 reward on failure


class PursuitEvasionEnvironment(Environment):
    """Pursuit-evasion environment with predators and prey."""
    
    def __init__(self, config: SwarmConfig) -> None:
        super().__init__(config)
        self.predator_ratio = config.environment_config.get("predator_ratio", 0.1)
        self.capture_radius = config.environment_config.get("capture_radius", 15.0)
        
    def reset(self) -> Dict[str, Any]:
        """Reset pursuit-evasion environment."""
        self.state = EnvironmentState()
        self.state.boundaries = {
            "min_x": 0,
            "max_x": self.config.arena_size[0],
            "min_y": 0,
            "max_y": self.config.arena_size[1]
        }
        
        return {
            "environment": "pursuit_evasion", 
            "bounds": self.state.boundaries,
            "predator_ratio": self.predator_ratio
        }
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict[str, Any], Dict[int, float], bool, Dict[str, Any]]:
        """Execute pursuit-evasion step."""
        self.state.step += 1
        rewards = {agent_id: 0.0 for agent_id in actions.keys()}
        
        # Basic survival reward for prey, hunting reward for predators handled in arena
        for agent_id in actions.keys():
            rewards[agent_id] = self.config.reward_config.get("survival_bonus", 0.01)
        
        done = self.state.step >= self.config.episode_length
        
        info = {
            "environment": "pursuit_evasion",
            "environment_stats": self.get_stats()
        }
        
        return {}, rewards, done, info