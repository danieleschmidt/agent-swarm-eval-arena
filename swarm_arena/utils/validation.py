"""Validation utilities for the Swarm Arena."""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
from ..exceptions import ValidationError


def validate_positive(value: Union[int, float], name: str) -> None:
    """Validate that a value is positive.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        
    Raises:
        ValidationError: If value is not positive
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_non_negative(value: Union[int, float], name: str) -> None:
    """Validate that a value is non-negative.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        
    Raises:
        ValidationError: If value is negative
    """
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")


def validate_range(value: Union[int, float], min_val: Union[int, float], 
                  max_val: Union[int, float], name: str) -> None:
    """Validate that a value is within a specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)  
        name: Name of the parameter for error messages
        
    Raises:
        ValidationError: If value is outside the range
    """
    if not (min_val <= value <= max_val):
        raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")


def validate_probability(value: float, name: str) -> None:
    """Validate that a value is a valid probability (0.0 to 1.0).
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        
    Raises:
        ValidationError: If value is not a valid probability
    """
    validate_range(value, 0.0, 1.0, name)


def validate_array_shape(array: np.ndarray, expected_shape: tuple, name: str) -> None:
    """Validate that an array has the expected shape.
    
    Args:
        array: Array to validate
        expected_shape: Expected shape tuple
        name: Name of the parameter for error messages
        
    Raises:
        ValidationError: If array shape doesn't match expected
    """
    if array.shape != expected_shape:
        raise ValidationError(f"{name} must have shape {expected_shape}, got {array.shape}")


def validate_position(position: np.ndarray, arena_size: tuple, name: str = "position") -> None:
    """Validate that a position is within arena bounds.
    
    Args:
        position: Position array [x, y]
        arena_size: Arena size (width, height)
        name: Name of the parameter for error messages
        
    Raises:
        ValidationError: If position is outside arena bounds
    """
    if len(position) != 2:
        raise ValidationError(f"{name} must be 2D, got {len(position)}D")
    
    if not (0 <= position[0] <= arena_size[0]):
        raise ValidationError(f"{name} x-coordinate must be between 0 and {arena_size[0]}, got {position[0]}")
    
    if not (0 <= position[1] <= arena_size[1]):
        raise ValidationError(f"{name} y-coordinate must be between 0 and {arena_size[1]}, got {position[1]}")


def validate_agent_id(agent_id: int, existing_agents: Dict[int, Any], name: str = "agent_id") -> None:
    """Validate agent ID uniqueness.
    
    Args:
        agent_id: Agent ID to validate
        existing_agents: Dictionary of existing agents
        name: Name of the parameter for error messages
        
    Raises:
        ValidationError: If agent ID is invalid or already exists
    """
    if not isinstance(agent_id, int):
        raise ValidationError(f"{name} must be an integer, got {type(agent_id)}")
    
    if agent_id < 0:
        raise ValidationError(f"{name} must be non-negative, got {agent_id}")
    
    if agent_id in existing_agents:
        raise ValidationError(f"{name} {agent_id} already exists")


def validate_action(action: int, valid_actions: Optional[List[int]] = None, name: str = "action") -> None:
    """Validate that an action is valid.
    
    Args:
        action: Action to validate
        valid_actions: List of valid action values (defaults to 0-5)
        name: Name of the parameter for error messages
        
    Raises:
        ValidationError: If action is invalid
    """
    if not isinstance(action, int):
        raise ValidationError(f"{name} must be an integer, got {type(action)}")
    
    if valid_actions is None:
        valid_actions = list(range(6))  # Default actions: 0-5
    
    if action not in valid_actions:
        raise ValidationError(f"{name} must be one of {valid_actions}, got {action}")


def validate_config_dict(config_dict: Dict[str, Any], required_keys: List[str], 
                        optional_keys: Optional[List[str]] = None) -> None:
    """Validate that a configuration dictionary has required keys.
    
    Args:
        config_dict: Configuration dictionary to validate
        required_keys: List of required keys
        optional_keys: List of optional keys (defaults to any additional keys allowed)
        
    Raises:
        ValidationError: If required keys are missing or invalid keys present
    """
    # Check required keys
    missing_keys = set(required_keys) - set(config_dict.keys())
    if missing_keys:
        raise ValidationError(f"Missing required configuration keys: {missing_keys}")
    
    # Check for invalid keys if optional_keys is specified
    if optional_keys is not None:
        valid_keys = set(required_keys) | set(optional_keys)
        invalid_keys = set(config_dict.keys()) - valid_keys
        if invalid_keys:
            raise ValidationError(f"Invalid configuration keys: {invalid_keys}")


def validate_type(value: Any, expected_type: type, name: str) -> None:
    """Validate that a value is of the expected type.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        name: Name of the parameter for error messages
        
    Raises:
        ValidationError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        raise ValidationError(f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}")


def validate_callable(func: Any, name: str) -> None:
    """Validate that a value is callable.
    
    Args:
        func: Function to validate
        name: Name of the parameter for error messages
        
    Raises:
        ValidationError: If value is not callable
    """
    if not callable(func):
        raise ValidationError(f"{name} must be callable, got {type(func).__name__}")


def validate_simulation_state(num_agents: int, positions: Dict[int, np.ndarray], 
                            velocities: Dict[int, np.ndarray], arena_size: tuple) -> None:
    """Validate the consistency of simulation state.
    
    Args:
        num_agents: Expected number of agents
        positions: Agent positions dictionary
        velocities: Agent velocities dictionary
        arena_size: Arena size tuple
        
    Raises:
        ValidationError: If simulation state is inconsistent
    """
    # Check dictionary lengths match
    if len(positions) != num_agents:
        raise ValidationError(f"Expected {num_agents} agent positions, got {len(positions)}")
    
    if len(velocities) != num_agents:
        raise ValidationError(f"Expected {num_agents} agent velocities, got {len(velocities)}")
    
    # Check agent IDs match
    if set(positions.keys()) != set(velocities.keys()):
        raise ValidationError("Agent IDs in positions and velocities don't match")
    
    # Validate individual positions and velocities
    for agent_id, pos in positions.items():
        validate_position(pos, arena_size, f"agent_{agent_id}_position")
        
        vel = velocities[agent_id]
        if len(vel) != 2:
            raise ValidationError(f"agent_{agent_id}_velocity must be 2D, got {len(vel)}D")


class StateValidator:
    """Comprehensive state validator for simulation integrity."""
    
    def __init__(self, config) -> None:
        """Initialize validator with configuration.
        
        Args:
            config: Swarm configuration object
        """
        self.config = config
        self.validation_errors: List[str] = []
    
    def validate_arena_state(self, arena) -> bool:
        """Validate complete arena state.
        
        Args:
            arena: Arena instance to validate
            
        Returns:
            True if valid, False otherwise
        """
        self.validation_errors.clear()
        
        try:
            # Validate basic state consistency
            validate_simulation_state(
                len(arena.agents),
                arena.agent_positions,
                arena.agent_velocities,
                self.config.arena_size
            )
            
            # Validate agent states
            for agent_id, agent in arena.agents.items():
                self._validate_agent_state(agent, agent_id)
            
            # Validate environment state
            self._validate_environment_state(arena.environment)
            
            return True
            
        except ValidationError as e:
            self.validation_errors.append(str(e))
            return False
    
    def _validate_agent_state(self, agent, agent_id: int) -> None:
        """Validate individual agent state."""
        if agent.state.health < 0 or agent.state.health > 1:
            raise ValidationError(f"Agent {agent_id} health must be between 0 and 1")
        
        if agent.state.energy < 0 or agent.state.energy > 1:
            raise ValidationError(f"Agent {agent_id} energy must be between 0 and 1")
        
        if agent.state.resources_collected < 0:
            raise ValidationError(f"Agent {agent_id} resources_collected must be non-negative")
    
    def _validate_environment_state(self, environment) -> None:
        """Validate environment state."""
        if environment.state.step < 0:
            raise ValidationError("Environment step must be non-negative")
        
        # Validate resources
        for i, resource in enumerate(environment.state.resources):
            validate_position(resource.position, self.config.arena_size, f"resource_{i}_position")
            
            if resource.value <= 0:
                raise ValidationError(f"Resource {i} value must be positive")
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors.
        
        Returns:
            List of validation error messages
        """
        return self.validation_errors.copy()