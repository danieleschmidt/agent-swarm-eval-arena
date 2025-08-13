"""Input validation and sanitization for security."""

import re
import json
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


@dataclass
class ValidationRule:
    """Single validation rule."""
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    required: bool = True


class ValidationError(Exception):
    """Raised when validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value


class InputSanitizer:
    """Sanitizes and validates input data."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        
        # Common regex patterns
        self.patterns = {
            'alphanumeric': re.compile(r'^[a-zA-Z0-9_]+$'),
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'filename': re.compile(r'^[a-zA-Z0-9._-]+$'),
            'safe_string': re.compile(r'^[a-zA-Z0-9\s.,!?()-]+$'),
            'numeric': re.compile(r'^[0-9.+-]+$'),
        }
        
        # Dangerous patterns to block
        self.dangerous_patterns = [
            re.compile(r'<script.*?>', re.IGNORECASE),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'exec\s*\(', re.IGNORECASE),
            re.compile(r'import\s+os', re.IGNORECASE),
            re.compile(r'subprocess', re.IGNORECASE),
        ]
    
    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """Sanitize string input.
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            raise ValidationError("Value must be a string", value=value)
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern.search(value):
                raise ValidationError(f"Potentially dangerous content detected", value=value)
        
        # HTML encode special characters
        value = (value
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#x27;'))
        
        return value.strip()
    
    def validate_numeric(self, 
                        value: Union[int, float], 
                        min_val: Optional[float] = None,
                        max_val: Optional[float] = None) -> Union[int, float]:
        """Validate numeric input.
        
        Args:
            value: Numeric value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Validated numeric value
        """
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValidationError("Value must be numeric", value=value)
        
        # Convert numpy types
        if isinstance(value, (np.integer, np.floating)):
            value = float(value)
        
        # Check for NaN or infinity
        if isinstance(value, float):
            if np.isnan(value):
                raise ValidationError("NaN values not allowed", value=value)
            if np.isinf(value):
                raise ValidationError("Infinite values not allowed", value=value)
        
        # Range validation
        if min_val is not None and value < min_val:
            raise ValidationError(f"Value must be >= {min_val}", value=value)
        
        if max_val is not None and value > max_val:
            raise ValidationError(f"Value must be <= {max_val}", value=value)
        
        return value
    
    def validate_list(self, 
                     value: List[Any], 
                     item_validator: Optional[Callable] = None,
                     max_length: int = 10000) -> List[Any]:
        """Validate list input.
        
        Args:
            value: List to validate
            item_validator: Validator function for individual items
            max_length: Maximum list length
            
        Returns:
            Validated list
        """
        if not isinstance(value, list):
            raise ValidationError("Value must be a list", value=value)
        
        if len(value) > max_length:
            raise ValidationError(f"List too long (max {max_length})", value=len(value))
        
        if item_validator:
            validated_items = []
            for i, item in enumerate(value):
                try:
                    validated_items.append(item_validator(item))
                except ValidationError as e:
                    raise ValidationError(f"List item {i}: {e}", value=item)
            return validated_items
        
        return value
    
    def validate_dict(self, 
                     value: Dict[str, Any],
                     allowed_keys: Optional[List[str]] = None,
                     required_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate dictionary input.
        
        Args:
            value: Dictionary to validate
            allowed_keys: List of allowed keys
            required_keys: List of required keys
            
        Returns:
            Validated dictionary
        """
        if not isinstance(value, dict):
            raise ValidationError("Value must be a dictionary", value=value)
        
        # Check required keys
        if required_keys:
            missing_keys = set(required_keys) - set(value.keys())
            if missing_keys:
                raise ValidationError(f"Missing required keys: {missing_keys}", value=value)
        
        # Check allowed keys
        if allowed_keys:
            extra_keys = set(value.keys()) - set(allowed_keys)
            if extra_keys:
                if self.validation_level == ValidationLevel.STRICT:
                    raise ValidationError(f"Unexpected keys: {extra_keys}", value=value)
                elif self.validation_level == ValidationLevel.MODERATE:
                    # Remove extra keys
                    value = {k: v for k, v in value.items() if k in allowed_keys}
        
        return value
    
    def validate_coordinates(self, coordinates: List[float]) -> List[float]:
        """Validate coordinate input.
        
        Args:
            coordinates: List of coordinates
            
        Returns:
            Validated coordinates
        """
        if not isinstance(coordinates, (list, tuple)):
            raise ValidationError("Coordinates must be a list or tuple", value=coordinates)
        
        if len(coordinates) != 2:
            raise ValidationError("Coordinates must have exactly 2 values", value=coordinates)
        
        validated = []
        for i, coord in enumerate(coordinates):
            validated.append(self.validate_numeric(
                coord, 
                min_val=-1e6, 
                max_val=1e6
            ))
        
        return validated
    
    def validate_agent_action(self, action: int) -> int:
        """Validate agent action.
        
        Args:
            action: Action integer
            
        Returns:
            Validated action
        """
        action = self.validate_numeric(action, min_val=0, max_val=10)
        return int(action)
    
    def validate_config_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration
        """
        allowed_keys = [
            'num_agents', 'arena_size', 'episode_length', 'max_agent_speed',
            'observation_radius', 'collision_detection', 'collision_radius',
            'seed', 'reward_config'
        ]
        
        config = self.validate_dict(config, allowed_keys=allowed_keys)
        
        # Validate specific fields
        if 'num_agents' in config:
            config['num_agents'] = int(self.validate_numeric(
                config['num_agents'], min_val=1, max_val=100000
            ))
        
        if 'arena_size' in config:
            config['arena_size'] = self.validate_list(
                config['arena_size'],
                item_validator=lambda x: self.validate_numeric(x, min_val=10, max_val=10000),
                max_length=2
            )
        
        if 'episode_length' in config:
            config['episode_length'] = int(self.validate_numeric(
                config['episode_length'], min_val=1, max_val=1000000
            ))
        
        if 'max_agent_speed' in config:
            config['max_agent_speed'] = self.validate_numeric(
                config['max_agent_speed'], min_val=0.1, max_val=1000
            )
        
        if 'observation_radius' in config:
            config['observation_radius'] = self.validate_numeric(
                config['observation_radius'], min_val=1, max_val=1000
            )
        
        if 'seed' in config and config['seed'] is not None:
            config['seed'] = int(self.validate_numeric(
                config['seed'], min_val=0, max_val=2**32 - 1
            ))
        
        return config
    
    def validate_filename(self, filename: str) -> str:
        """Validate filename for safety.
        
        Args:
            filename: Filename to validate
            
        Returns:
            Validated filename
        """
        filename = self.sanitize_string(filename, max_length=255)
        
        # Check filename pattern
        if not self.patterns['filename'].match(filename):
            raise ValidationError("Invalid filename format", value=filename)
        
        # Block dangerous filenames
        dangerous_names = [
            'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4',
            'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2',
            'lpt3', 'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'
        ]
        
        if filename.lower() in dangerous_names:
            raise ValidationError("Reserved filename", value=filename)
        
        # Block path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            raise ValidationError("Path traversal not allowed", value=filename)
        
        return filename
    
    def validate_email(self, email: str) -> str:
        """Validate email address.
        
        Args:
            email: Email to validate
            
        Returns:
            Validated email
        """
        email = self.sanitize_string(email, max_length=254).lower()
        
        if not self.patterns['email'].match(email):
            raise ValidationError("Invalid email format", value=email)
        
        return email
    
    def validate_json_input(self, json_str: str, max_size: int = 1024*1024) -> Dict[str, Any]:
        """Validate JSON input.
        
        Args:
            json_str: JSON string to validate
            max_size: Maximum JSON size in bytes
            
        Returns:
            Parsed JSON data
        """
        if len(json_str.encode('utf-8')) > max_size:
            raise ValidationError(f"JSON too large (max {max_size} bytes)")
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}")
        
        # Limit nesting depth
        def check_depth(obj, depth=0, max_depth=10):
            if depth > max_depth:
                raise ValidationError("JSON nesting too deep")
            
            if isinstance(obj, dict):
                for value in obj.values():
                    check_depth(value, depth + 1, max_depth)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, depth + 1, max_depth)
        
        check_depth(data)
        return data


class ConfigValidator:
    """Validates swarm arena configurations."""
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        
        # Define validation rules
        self.rules = {
            'num_agents': ValidationRule(
                name='num_agents',
                validator=lambda x: isinstance(x, int) and 1 <= x <= 100000,
                error_message="num_agents must be integer between 1 and 100000"
            ),
            'arena_size': ValidationRule(
                name='arena_size',
                validator=lambda x: (isinstance(x, (list, tuple)) and 
                                   len(x) == 2 and 
                                   all(isinstance(v, (int, float)) and v >= 10 for v in x)),
                error_message="arena_size must be [width, height] with values >= 10"
            ),
            'episode_length': ValidationRule(
                name='episode_length',
                validator=lambda x: isinstance(x, int) and 1 <= x <= 1000000,
                error_message="episode_length must be integer between 1 and 1000000"
            ),
            'max_agent_speed': ValidationRule(
                name='max_agent_speed',
                validator=lambda x: isinstance(x, (int, float)) and 0.1 <= x <= 1000,
                error_message="max_agent_speed must be number between 0.1 and 1000"
            ),
            'observation_radius': ValidationRule(
                name='observation_radius',
                validator=lambda x: isinstance(x, (int, float)) and 1 <= x <= 1000,
                error_message="observation_radius must be number between 1 and 1000"
            ),
            'collision_detection': ValidationRule(
                name='collision_detection',
                validator=lambda x: isinstance(x, bool),
                error_message="collision_detection must be boolean",
                required=False
            ),
            'collision_radius': ValidationRule(
                name='collision_radius',
                validator=lambda x: isinstance(x, (int, float)) and 0.1 <= x <= 100,
                error_message="collision_radius must be number between 0.1 and 100",
                required=False
            ),
            'seed': ValidationRule(
                name='seed',
                validator=lambda x: x is None or (isinstance(x, int) and 0 <= x <= 2**32 - 1),
                error_message="seed must be None or integer between 0 and 2^32-1",
                required=False
            )
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration
        """
        # First sanitize the dictionary
        config = self.sanitizer.validate_config_dict(config)
        
        # Apply validation rules
        errors = []
        
        for field_name, rule in self.rules.items():
            if field_name in config:
                value = config[field_name]
                if not rule.validator(value):
                    errors.append(f"{field_name}: {rule.error_message}")
            elif rule.required:
                errors.append(f"{field_name}: Required field missing")
        
        if errors:
            raise ValidationError(f"Configuration validation failed: {'; '.join(errors)}")
        
        # Additional cross-field validation
        self._validate_config_consistency(config)
        
        return config
    
    def _validate_config_consistency(self, config: Dict[str, Any]) -> None:
        """Validate consistency between configuration fields.
        
        Args:
            config: Configuration dictionary
        """
        errors = []
        
        # Arena size vs agent count
        if 'arena_size' in config and 'num_agents' in config:
            arena_area = config['arena_size'][0] * config['arena_size'][1]
            agent_density = config['num_agents'] / arena_area
            
            if agent_density > 0.01:  # More than 1 agent per 100 square units
                errors.append("Agent density too high - may cause performance issues")
        
        # Collision radius vs observation radius
        if ('collision_radius' in config and 
            'observation_radius' in config and
            config['collision_radius'] > config['observation_radius']):
            errors.append("collision_radius should not exceed observation_radius")
        
        # Agent speed vs arena size
        if ('max_agent_speed' in config and 
            'arena_size' in config and
            'episode_length' in config):
            max_distance = config['max_agent_speed'] * config['episode_length']
            arena_diagonal = (config['arena_size'][0]**2 + config['arena_size'][1]**2)**0.5
            
            if max_distance > arena_diagonal * 10:
                errors.append("max_agent_speed may be too high for arena size and episode length")
        
        if errors:
            raise ValidationError(f"Configuration consistency issues: {'; '.join(errors)}")


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.limits = {
            'default': (100, 3600),  # 100 requests per hour
            'experiment': (10, 300),  # 10 experiments per 5 minutes
            'simulation': (50, 1800),  # 50 simulations per 30 minutes
        }
    
    def is_allowed(self, 
                   identifier: str, 
                   limit_type: str = 'default') -> bool:
        """Check if request is allowed under rate limit.
        
        Args:
            identifier: Unique identifier (IP, user, etc.)
            limit_type: Type of rate limit
            
        Returns:
            True if request is allowed
        """
        if limit_type not in self.limits:
            limit_type = 'default'
        
        max_requests, time_window = self.limits[limit_type]
        current_time = time.time()
        
        # Initialize or clean up old requests
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove requests outside time window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time <= time_window
        ]
        
        # Check if under limit
        if len(self.requests[identifier]) >= max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(current_time)
        return True
    
    def get_remaining_requests(self, 
                              identifier: str, 
                              limit_type: str = 'default') -> int:
        """Get remaining requests for identifier.
        
        Args:
            identifier: Unique identifier
            limit_type: Type of rate limit
            
        Returns:
            Number of remaining requests
        """
        if limit_type not in self.limits:
            limit_type = 'default'
        
        max_requests, _ = self.limits[limit_type]
        current_requests = len(self.requests.get(identifier, []))
        
        return max(0, max_requests - current_requests)