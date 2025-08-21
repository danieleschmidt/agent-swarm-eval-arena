"""Configuration classes for the Swarm Arena."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ..exceptions import ConfigurationError
from ..utils.validation import (
    validate_positive, validate_non_negative, validate_probability, validate_range
)


@dataclass
class SwarmConfig:
    """Configuration for swarm arena simulation.
    
    Attributes:
        num_agents: Number of agents in the simulation
        arena_size: (width, height) dimensions of the arena
        episode_length: Maximum steps per episode
        resource_spawn_rate: Probability of spawning resource per step
        physics_engine: Physics engine to use ('simple', 'mujoco')
        collision_detection: Enable collision detection between agents
        communication_enabled: Allow agents to communicate
        communication_radius: Maximum distance for agent communication
        seed: Random seed for reproducibility
        render_mode: Rendering mode ('none', 'human', 'rgb_array')
        max_fps: Maximum frames per second for rendering
        observation_radius: Agent observation radius
        action_space_type: Type of action space ('discrete', 'continuous')
        reward_config: Reward function configuration
    """
    
    # Core simulation parameters
    num_agents: int = 100
    arena_size: Tuple[float, float] = (1000.0, 1000.0)
    episode_length: int = 1000
    resource_spawn_rate: float = 0.1
    
    # Physics and collision
    physics_engine: str = "simple"  # 'simple' or 'mujoco'
    collision_detection: bool = True
    collision_radius: float = 5.0
    
    # Communication
    communication_enabled: bool = False
    enable_communication: bool = False  # Alternative name for compatibility
    communication_radius: float = 50.0
    max_messages_per_step: int = 10
    
    # Reproducibility
    seed: Optional[int] = None
    
    # Rendering
    render_mode: str = "none"  # 'none', 'human', 'rgb_array'
    max_fps: int = 60
    
    # Agent capabilities
    observation_radius: float = 100.0
    action_space_type: str = "discrete"  # 'discrete' or 'continuous'
    max_agent_speed: float = 10.0
    
    # Reward configuration
    reward_config: Dict[str, Any] = field(default_factory=lambda: {
        "resource_collection": 1.0,
        "survival_bonus": 0.01,
        "collision_penalty": -0.1,
        "time_penalty": -0.001,
        "cooperation_bonus": 0.5,
    })
    
    # Environment specific
    environment_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        try:
            # Validate core parameters
            validate_positive(self.num_agents, "num_agents")
            validate_positive(self.episode_length, "episode_length")
            validate_probability(self.resource_spawn_rate, "resource_spawn_rate")
            
            # Validate arena size
            if len(self.arena_size) != 2:
                raise ConfigurationError("arena_size must be a 2-tuple")
            
            for i, dim in enumerate(self.arena_size):
                validate_positive(dim, f"arena_size[{i}]")
            
            # Validate physics parameters
            if self.physics_engine not in ["simple", "mujoco"]:
                raise ConfigurationError("physics_engine must be 'simple' or 'mujoco'")
            
            validate_positive(self.collision_radius, "collision_radius")
            validate_positive(self.communication_radius, "communication_radius")
            validate_positive(self.max_messages_per_step, "max_messages_per_step")
            
            # Validate rendering parameters
            if self.render_mode not in ["none", "human", "rgb_array"]:
                raise ConfigurationError("render_mode must be 'none', 'human', or 'rgb_array'")
            
            validate_positive(self.max_fps, "max_fps")
            
            # Validate agent parameters
            if self.action_space_type not in ["discrete", "continuous"]:
                raise ConfigurationError("action_space_type must be 'discrete' or 'continuous'")
            
            validate_positive(self.observation_radius, "observation_radius")
            validate_positive(self.max_agent_speed, "max_agent_speed")
            
            # Validate seed if provided
            if self.seed is not None:
                if not isinstance(self.seed, int) or self.seed < 0:
                    raise ConfigurationError("seed must be a non-negative integer")
            
            # Validate reward configuration
            self._validate_reward_config()
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "num_agents": self.num_agents,
            "arena_size": self.arena_size,
            "episode_length": self.episode_length,
            "resource_spawn_rate": self.resource_spawn_rate,
            "physics_engine": self.physics_engine,
            "collision_detection": self.collision_detection,
            "collision_radius": self.collision_radius,
            "communication_enabled": self.communication_enabled,
            "communication_radius": self.communication_radius,
            "max_messages_per_step": self.max_messages_per_step,
            "seed": self.seed,
            "render_mode": self.render_mode,
            "max_fps": self.max_fps,
            "observation_radius": self.observation_radius,
            "action_space_type": self.action_space_type,
            "max_agent_speed": self.max_agent_speed,
            "reward_config": self.reward_config,
            "environment_config": self.environment_config,
        }
    
    def _validate_reward_config(self) -> None:
        """Validate reward configuration structure."""
        required_keys = [
            "resource_collection", "survival_bonus", "collision_penalty", 
            "time_penalty", "cooperation_bonus"
        ]
        
        for key in required_keys:
            if key not in self.reward_config:
                raise ConfigurationError(f"Missing required reward config key: {key}")
            
            value = self.reward_config[key]
            if not isinstance(value, (int, float)):
                raise ConfigurationError(f"Reward config '{key}' must be numeric, got {type(value)}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SwarmConfig":
        """Create configuration from dictionary."""
        try:
            return cls(**config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to create config from dict: {str(e)}")
    
    def copy(self, **kwargs: Any) -> "SwarmConfig":
        """Create a copy with optional parameter overrides."""
        try:
            config_dict = self.to_dict()
            config_dict.update(kwargs)
            return self.from_dict(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to copy config: {str(e)}")