"""Tests for validation functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from swarm_arena.validation.input_validator import InputValidator
from swarm_arena.core.config import SwarmConfig
from swarm_arena.core.arena import Arena
from swarm_arena.core.agent import RandomAgent
from swarm_arena.exceptions import ValidationError


class TestInputValidator:
    """Test input validation functionality."""
    
    def test_validator_initialization(self):
        """Test validator can be initialized."""
        validator = InputValidator()
        assert validator is not None
    
    def test_validate_positive_number(self):
        """Test positive number validation."""
        validator = InputValidator()
        
        # Valid positive numbers
        assert validator.validate_positive_number(1) is True
        assert validator.validate_positive_number(0.1) is True
        assert validator.validate_positive_number(1000) is True
        
        # Invalid values
        assert validator.validate_positive_number(0) is False
        assert validator.validate_positive_number(-1) is False
        assert validator.validate_positive_number(-0.1) is False
    
    def test_validate_range(self):
        """Test range validation."""
        validator = InputValidator()
        
        # Valid range values
        assert validator.validate_range(5, 1, 10) is True
        assert validator.validate_range(1, 1, 10) is True
        assert validator.validate_range(10, 1, 10) is True
        
        # Invalid range values
        assert validator.validate_range(0, 1, 10) is False
        assert validator.validate_range(11, 1, 10) is False
        assert validator.validate_range(-1, 1, 10) is False
    
    def test_validate_swarm_config(self):
        """Test SwarmConfig validation."""
        validator = InputValidator()
        
        # Valid config
        valid_config = SwarmConfig(
            num_agents=10,
            episode_length=100,
            arena_size=(1000, 1000),
            seed=42
        )
        assert validator.validate_swarm_config(valid_config) is True
        
        # Invalid config - negative agents
        invalid_config = SwarmConfig(
            num_agents=-5,
            episode_length=100,
            arena_size=(1000, 1000)
        )
        assert validator.validate_swarm_config(invalid_config) is False
        
        # Invalid config - zero episode length
        invalid_config2 = SwarmConfig(
            num_agents=10,
            episode_length=0,
            arena_size=(1000, 1000)
        )
        assert validator.validate_swarm_config(invalid_config2) is False
    
    def test_validate_arena_state(self):
        """Test arena state validation."""
        validator = InputValidator()
        
        # Create a valid arena
        config = SwarmConfig(num_agents=5, episode_length=50, seed=42)
        arena = Arena(config)
        
        # Add agents
        for i in range(5):
            arena.agents[i] = RandomAgent()
        
        # Valid arena state
        result = validator.validate_arena_state(arena)
        assert result is True
        
        # Check validation errors are empty
        errors = validator.get_validation_errors()
        assert len(errors) == 0
    
    def test_validate_agent_positions(self):
        """Test agent position validation."""
        validator = InputValidator()
        
        # Valid positions
        positions = {
            0: np.array([10.0, 20.0]),
            1: np.array([50.0, 60.0])
        }
        arena_size = (100, 100)
        
        assert validator.validate_agent_positions(positions, arena_size) is True
        
        # Invalid positions - out of bounds
        invalid_positions = {
            0: np.array([150.0, 20.0]),  # x out of bounds
            1: np.array([50.0, 120.0])   # y out of bounds
        }
        
        assert validator.validate_agent_positions(invalid_positions, arena_size) is False
        
        # Check error message
        errors = validator.get_validation_errors()
        assert len(errors) > 0
        assert any("out of bounds" in error.lower() for error in errors)
    
    def test_validate_simulation_step(self):
        """Test simulation step validation."""
        validator = InputValidator()
        
        # Create arena and run validation during simulation
        config = SwarmConfig(num_agents=3, episode_length=10, seed=42)
        arena = Arena(config)
        
        # Add agents
        for i in range(3):
            arena.agents[i] = RandomAgent()
        
        # Validate initial state
        initial_validation = validator.validate_arena_state(arena)
        assert initial_validation is True
        
        # Run a few steps and validate state after each
        for step in range(5):
            # Simulate a step (simplified)
            arena.current_step = step
            step_validation = validator.validate_arena_state(arena)
            assert step_validation is True
    
    def test_error_accumulation(self):
        """Test that validation errors accumulate properly."""
        validator = InputValidator()
        
        # Generate multiple errors
        validator.validate_positive_number(-1)
        validator.validate_range(15, 1, 10)
        
        errors = validator.get_validation_errors()
        assert len(errors) >= 2  # Should have accumulated errors
        
        # Clear errors
        validator.clear_errors()
        errors = validator.get_validation_errors()
        assert len(errors) == 0
    
    def test_custom_validation_rules(self):
        """Test custom validation rules."""
        validator = InputValidator()
        
        # Test custom rule for agent count vs arena size ratio
        def validate_agent_density(num_agents: int, arena_size: tuple) -> bool:
            arena_area = arena_size[0] * arena_size[1]
            density = num_agents / arena_area
            return density < 0.001  # Max 1 agent per 1000 sq units
        
        # Valid density
        assert validate_agent_density(10, (1000, 1000)) is True
        
        # Invalid density (too crowded)
        assert validate_agent_density(1000, (100, 100)) is False
    
    def test_batch_validation(self):
        """Test validating multiple items at once."""
        validator = InputValidator()
        
        configs = [
            SwarmConfig(num_agents=10, episode_length=100),
            SwarmConfig(num_agents=20, episode_length=200),
            SwarmConfig(num_agents=-5, episode_length=50),  # Invalid
        ]
        
        results = [validator.validate_swarm_config(config) for config in configs]
        
        assert results[0] is True
        assert results[1] is True
        assert results[2] is False
        
        # Should have accumulated errors from invalid config
        errors = validator.get_validation_errors()
        assert len(errors) > 0


class TestValidationIntegration:
    """Integration tests for validation with other components."""
    
    def test_arena_with_validation(self):
        """Test arena creation with validation enabled."""
        config = SwarmConfig(num_agents=5, episode_length=20, seed=42)
        arena = Arena(config)
        validator = InputValidator()
        
        # Add agents
        for i in range(5):
            arena.agents[i] = RandomAgent()
        
        # Validate arena can be created and run
        assert validator.validate_arena_state(arena) is True
        
        # Run simulation with validation
        for step in range(10):
            # Simulate step
            arena.current_step = step
            
            # Validate state after each step
            is_valid = validator.validate_arena_state(arena)
            assert is_valid is True, f"Validation failed at step {step}: {validator.get_validation_errors()}"
    
    def test_config_validation_prevents_errors(self):
        """Test that config validation prevents runtime errors."""
        validator = InputValidator()
        
        # Invalid config that should be caught
        invalid_config = SwarmConfig(
            num_agents=0,  # Invalid - no agents
            episode_length=-10,  # Invalid - negative length
            arena_size=(0, 0),  # Invalid - zero size
        )
        
        # Validation should catch these errors
        is_valid = validator.validate_swarm_config(invalid_config)
        assert is_valid is False
        
        errors = validator.get_validation_errors()
        assert len(errors) > 0
        
        # Verify specific error types are caught
        error_text = " ".join(errors).lower()
        assert "agent" in error_text or "episode" in error_text or "arena" in error_text
    
    def test_validation_with_edge_cases(self):
        """Test validation with edge cases."""
        validator = InputValidator()
        
        # Edge case: single agent
        single_agent_config = SwarmConfig(num_agents=1, episode_length=1)
        assert validator.validate_swarm_config(single_agent_config) is True
        
        # Edge case: very large arena
        large_arena_config = SwarmConfig(
            num_agents=10,
            episode_length=100,
            arena_size=(100000, 100000)
        )
        assert validator.validate_swarm_config(large_arena_config) is True
        
        # Edge case: maximum reasonable values
        max_config = SwarmConfig(
            num_agents=10000,
            episode_length=100000,
            arena_size=(50000, 50000)
        )
        # This might be valid or invalid depending on system constraints
        result = validator.validate_swarm_config(max_config)
        # Just ensure it doesn't crash
        assert isinstance(result, bool)