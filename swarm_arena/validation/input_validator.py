"""Input validation and sanitization for Swarm Arena."""

import re
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..exceptions import ValidationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationRule:
    """Represents a validation rule."""
    field_name: str
    rule_type: str
    parameters: Dict[str, Any]
    error_message: str


class InputValidator:
    """Comprehensive input validation for all user inputs."""
    
    # Common validation patterns
    PATTERNS = {
        "agent_id": re.compile(r"^[a-zA-Z0-9_-]+$"),
        "file_name": re.compile(r"^[a-zA-Z0-9_.-]+$"),
        "config_key": re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$"),
        "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
    }
    
    # Safe file extensions
    SAFE_EXTENSIONS = {".json", ".txt", ".csv", ".log", ".md", ".yaml", ".yml"}
    
    def __init__(self):
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default validation rules."""
        # Configuration validation rules
        self.add_rule("num_agents", "integer", {"min": 1, "max": 10000}, 
                     "Number of agents must be between 1 and 10000")
        
        self.add_rule("arena_size", "tuple", {"length": 2, "element_type": "float", "min": 10, "max": 50000},
                     "Arena size must be a 2-tuple of floats between 10 and 50000")
        
        self.add_rule("episode_length", "integer", {"min": 1, "max": 100000},
                     "Episode length must be between 1 and 100000")
        
        self.add_rule("resource_spawn_rate", "float", {"min": 0.0, "max": 1.0},
                     "Resource spawn rate must be between 0.0 and 1.0")
        
        self.add_rule("seed", "integer", {"min": 0, "max": 2**31-1},
                     "Seed must be a non-negative integer")
        
        # Agent validation rules
        self.add_rule("agent_type", "string", {"choices": ["default", "cooperative", "competitive", "random", "learning", "hierarchical", "swarm", "adaptive"]},
                     "Invalid agent type")
        
        self.add_rule("position", "list", {"length": 2, "element_type": "float"},
                     "Position must be a 2-element list of floats")
        
        # File validation rules
        self.add_rule("file_path", "file_path", {"extensions": list(self.SAFE_EXTENSIONS)},
                     "Invalid file path or unsafe extension")
    
    def add_rule(self, field_name: str, rule_type: str, parameters: Dict[str, Any], error_message: str) -> None:
        """Add a validation rule."""
        if field_name not in self.validation_rules:
            self.validation_rules[field_name] = []
        
        rule = ValidationRule(field_name, rule_type, parameters, error_message)
        self.validation_rules[field_name].append(rule)
    
    def validate_field(self, field_name: str, value: Any) -> bool:
        """Validate a single field."""
        if field_name not in self.validation_rules:
            return True  # No rules defined, assume valid
        
        for rule in self.validation_rules[field_name]:
            try:
                if not self._apply_rule(value, rule):
                    raise ValidationError(f"{field_name}: {rule.error_message}")
            except Exception as e:
                if isinstance(e, ValidationError):
                    raise
                raise ValidationError(f"{field_name}: Validation error - {str(e)}")
        
        return True
    
    def validate_dict(self, data: Dict[str, Any], required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate a dictionary of values."""
        validated_data = {}
        
        # Check required fields
        if required_fields:
            for field in required_fields:
                if field not in data:
                    raise ValidationError(f"Required field missing: {field}")
        
        # Validate each field
        for field_name, value in data.items():
            self.validate_field(field_name, value)
            validated_data[field_name] = self._sanitize_value(value)
        
        return validated_data
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a complete configuration dictionary."""
        try:
            # Required fields for basic config
            required_fields = ["num_agents", "arena_size", "episode_length"]
            
            # Validate basic structure
            validated_config = self.validate_dict(config, required_fields)
            
            # Additional configuration-specific validations
            self._validate_config_consistency(validated_config)
            
            logger.debug(f"Configuration validation successful")
            return validated_config
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Configuration validation failed: {str(e)}")
    
    def validate_agent_config(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent configuration."""
        try:
            required_fields = ["agent_type"]
            validated_config = self.validate_dict(agent_config, required_fields)
            
            # Validate agent-specific parameters
            agent_type = validated_config["agent_type"]
            if agent_type == "learning":
                self._validate_learning_agent_config(validated_config)
            elif agent_type == "hierarchical":
                self._validate_hierarchical_agent_config(validated_config)
            
            return validated_config
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Agent configuration validation failed: {str(e)}")
    
    def validate_file_upload(self, file_path: str, content: bytes) -> bool:
        """Validate file upload for security."""
        try:
            # Validate file path
            self.validate_field("file_path", file_path)
            
            # Check file size (max 100MB)
            if len(content) > 100 * 1024 * 1024:
                raise ValidationError("File too large (max 100MB)")
            
            # Check for executable content in file headers
            if self._contains_executable_content(content):
                raise ValidationError("File contains potentially dangerous content")
            
            # Validate JSON files
            if file_path.endswith(".json"):
                try:
                    json.loads(content.decode('utf-8'))
                except json.JSONDecodeError as e:
                    raise ValidationError(f"Invalid JSON format: {str(e)}")
            
            return True
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"File validation failed: {str(e)}")
    
    def _apply_rule(self, value: Any, rule: ValidationRule) -> bool:
        """Apply a specific validation rule."""
        rule_type = rule.rule_type
        params = rule.parameters
        
        if rule_type == "integer":
            return self._validate_integer(value, params)
        elif rule_type == "float":
            return self._validate_float(value, params)
        elif rule_type == "string":
            return self._validate_string(value, params)
        elif rule_type == "list":
            return self._validate_list(value, params)
        elif rule_type == "tuple":
            return self._validate_tuple(value, params)
        elif rule_type == "file_path":
            return self._validate_file_path(value, params)
        elif rule_type == "pattern":
            return self._validate_pattern(value, params)
        else:
            logger.warning(f"Unknown validation rule type: {rule_type}")
            return True
    
    def _validate_integer(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate integer value."""
        if not isinstance(value, int):
            return False
        
        if "min" in params and value < params["min"]:
            return False
        
        if "max" in params and value > params["max"]:
            return False
        
        if "choices" in params and value not in params["choices"]:
            return False
        
        return True
    
    def _validate_float(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate float value."""
        if not isinstance(value, (int, float)):
            return False
        
        value = float(value)
        
        if "min" in params and value < params["min"]:
            return False
        
        if "max" in params and value > params["max"]:
            return False
        
        return True
    
    def _validate_string(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate string value."""
        if not isinstance(value, str):
            return False
        
        if "min_length" in params and len(value) < params["min_length"]:
            return False
        
        if "max_length" in params and len(value) > params["max_length"]:
            return False
        
        if "choices" in params and value not in params["choices"]:
            return False
        
        if "pattern" in params:
            pattern = re.compile(params["pattern"])
            if not pattern.match(value):
                return False
        
        return True
    
    def _validate_list(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate list value."""
        if not isinstance(value, list):
            return False
        
        if "length" in params and len(value) != params["length"]:
            return False
        
        if "min_length" in params and len(value) < params["min_length"]:
            return False
        
        if "max_length" in params and len(value) > params["max_length"]:
            return False
        
        if "element_type" in params:
            element_type = params["element_type"]
            for item in value:
                if element_type == "float" and not isinstance(item, (int, float)):
                    return False
                elif element_type == "int" and not isinstance(item, int):
                    return False
                elif element_type == "string" and not isinstance(item, str):
                    return False
        
        return True
    
    def _validate_tuple(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate tuple value."""
        if not isinstance(value, (tuple, list)):
            return False
        
        # Convert to list for uniform handling
        value = list(value)
        
        if "length" in params and len(value) != params["length"]:
            return False
        
        if "element_type" in params:
            element_type = params["element_type"]
            for item in value:
                if element_type == "float" and not isinstance(item, (int, float)):
                    return False
                elif element_type == "int" and not isinstance(item, int):
                    return False
                elif element_type == "string" and not isinstance(item, str):
                    return False
        
        # Additional range checks for numeric tuples
        if "min" in params or "max" in params:
            for item in value:
                if "min" in params and item < params["min"]:
                    return False
                if "max" in params and item > params["max"]:
                    return False
        
        return True
    
    def _validate_file_path(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate file path."""
        if not isinstance(value, str):
            return False
        
        # Check for path traversal
        if ".." in value or value.startswith("/") or "\\" in value:
            return False
        
        # Check extension
        if "extensions" in params:
            path = Path(value)
            if path.suffix not in params["extensions"]:
                return False
        
        # Check filename pattern
        if not self.PATTERNS["file_name"].match(Path(value).name):
            return False
        
        return True
    
    def _validate_pattern(self, value: Any, params: Dict[str, Any]) -> bool:
        """Validate against regex pattern."""
        if not isinstance(value, str):
            return False
        
        pattern_name = params.get("pattern_name")
        if pattern_name and pattern_name in self.PATTERNS:
            return bool(self.PATTERNS[pattern_name].match(value))
        
        pattern_str = params.get("pattern")
        if pattern_str:
            pattern = re.compile(pattern_str)
            return bool(pattern.match(value))
        
        return True
    
    def _validate_config_consistency(self, config: Dict[str, Any]) -> None:
        """Validate internal consistency of configuration."""
        # Check arena size vs agent count
        arena_size = config.get("arena_size", (1000, 1000))
        num_agents = config.get("num_agents", 100)
        
        arena_area = arena_size[0] * arena_size[1]
        agent_density = num_agents / arena_area
        
        if agent_density > 0.01:  # More than 1 agent per 100 square units
            logger.warning(f"High agent density: {agent_density:.4f} agents per square unit")
        
        # Check observation radius vs arena size
        observation_radius = config.get("observation_radius", 100)
        min_arena_dim = min(arena_size)
        
        if observation_radius > min_arena_dim / 2:
            logger.warning(f"Observation radius ({observation_radius}) is large relative to arena size")
    
    def _validate_learning_agent_config(self, config: Dict[str, Any]) -> None:
        """Validate learning agent specific configuration."""
        learning_rate = config.get("learning_rate", 0.1)
        if not (0.001 <= learning_rate <= 1.0):
            raise ValidationError("Learning rate must be between 0.001 and 1.0")
        
        epsilon = config.get("epsilon", 0.1)
        if not (0.0 <= epsilon <= 1.0):
            raise ValidationError("Epsilon must be between 0.0 and 1.0")
    
    def _validate_hierarchical_agent_config(self, config: Dict[str, Any]) -> None:
        """Validate hierarchical agent specific configuration."""
        strategy_duration = config.get("strategy_duration", 100)
        if not (10 <= strategy_duration <= 10000):
            raise ValidationError("Strategy duration must be between 10 and 10000")
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a value for safe storage/processing."""
        if isinstance(value, str):
            # Remove null bytes and control characters
            value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', value)
            # Limit string length
            value = value[:1000]
        
        return value
    
    def _contains_executable_content(self, content: bytes) -> bool:
        """Check if content contains executable file signatures."""
        # Check for common executable file headers
        executable_signatures = [
            b'\x4d\x5a',  # PE executable (Windows)
            b'\x7f\x45\x4c\x46',  # ELF executable (Linux)
            b'\xfe\xed\xfa\xce',  # Mach-O executable (macOS)
            b'\xcf\xfa\xed\xfe',  # Mach-O executable (macOS)
            b'#!/bin/',  # Shell script
            b'#!/usr/bin/',  # Shell script
        ]
        
        for signature in executable_signatures:
            if content.startswith(signature):
                return True
        
        return False


# Global validator instance
input_validator = InputValidator()