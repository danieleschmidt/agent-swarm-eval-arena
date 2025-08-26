"""Input sanitization and validation for secure agent operations."""

import re
import numpy as np
from typing import Any, Dict, List, Union, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Validation rule for input sanitization."""
    name: str
    validator: callable
    error_message: str
    severity: str = "error"  # error, warning, info


class InputSanitizer:
    """Comprehensive input sanitization and validation system."""
    
    def __init__(self):
        """Initialize input sanitizer with default rules."""
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.sanitization_stats = {
            'total_checks': 0,
            'violations_detected': 0,
            'blocked_inputs': 0,
            'sanitized_inputs': 0
        }
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """Register default validation rules."""
        
        # Agent configuration rules
        self.add_rule('agent_config', ValidationRule(
            name='positive_numeric',
            validator=lambda x: isinstance(x, (int, float)) and x >= 0,
            error_message="Agent configuration values must be non-negative numbers",
            severity="error"
        ))
        
        self.add_rule('agent_config', ValidationRule(
            name='finite_values',
            validator=lambda x: isinstance(x, (int, float)) and np.isfinite(x),
            error_message="Agent configuration values must be finite",
            severity="error"
        ))
        
        # Position validation rules
        self.add_rule('position', ValidationRule(
            name='valid_coordinates',
            validator=self._validate_position_coordinates,
            error_message="Position must be 2D coordinates with finite values",
            severity="error"
        ))
        
        self.add_rule('position', ValidationRule(
            name='reasonable_bounds',
            validator=self._validate_position_bounds,
            error_message="Position coordinates exceed reasonable bounds",
            severity="warning"
        ))
        
        # Action validation rules
        self.add_rule('action', ValidationRule(
            name='valid_action_type',
            validator=lambda x: isinstance(x, int) and 0 <= x <= 5,
            error_message="Action must be integer between 0-5",
            severity="error"
        ))
        
        # String input rules (prevent injection attacks)
        self.add_rule('string_input', ValidationRule(
            name='no_script_injection',
            validator=self._validate_no_script_injection,
            error_message="String contains potentially malicious script content",
            severity="error"
        ))
        
        self.add_rule('string_input', ValidationRule(
            name='reasonable_length',
            validator=lambda x: isinstance(x, str) and len(x) <= 1000,
            error_message="String input exceeds maximum length",
            severity="error"
        ))
        
        # Configuration rules
        self.add_rule('config', ValidationRule(
            name='no_dangerous_paths',
            validator=self._validate_no_dangerous_paths,
            error_message="Configuration contains potentially dangerous file paths",
            severity="error"
        ))
    
    def add_rule(self, category: str, rule: ValidationRule) -> None:
        """Add validation rule for a category."""
        if category not in self.rules:
            self.rules[category] = []
        self.rules[category].append(rule)
    
    def sanitize_agent_config(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Sanitize agent configuration dictionary."""
        sanitized = {}
        warnings = []
        
        for key, value in config.items():
            # Sanitize key name
            safe_key = self._sanitize_key_name(key)
            if safe_key != key:
                warnings.append(f"Config key '{key}' sanitized to '{safe_key}'")
            
            # Validate and sanitize value
            try:
                safe_value, value_warnings = self.validate_input('agent_config', value)
                sanitized[safe_key] = safe_value
                warnings.extend(value_warnings)
            except ValueError as e:
                warnings.append(f"Config value for '{key}' failed validation: {e}")
                # Use safe default
                sanitized[safe_key] = self._get_safe_default(value)
        
        self.sanitization_stats['sanitized_inputs'] += 1
        return sanitized, warnings
    
    def sanitize_position(self, position: Union[List, Tuple, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """Sanitize position coordinates."""
        warnings = []
        
        try:
            # Convert to numpy array
            pos_array = np.array(position, dtype=np.float32)
            
            # Validate
            sanitized_pos, pos_warnings = self.validate_input('position', pos_array)
            warnings.extend(pos_warnings)
            
            # Ensure finite values
            if not np.all(np.isfinite(sanitized_pos)):
                warnings.append("Position contained non-finite values, replaced with zeros")
                sanitized_pos = np.nan_to_num(sanitized_pos, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clamp to reasonable bounds
            max_coord = 1000000  # 1M units
            if np.any(np.abs(sanitized_pos) > max_coord):
                warnings.append("Position coordinates clamped to reasonable bounds")
                sanitized_pos = np.clip(sanitized_pos, -max_coord, max_coord)
            
            self.sanitization_stats['sanitized_inputs'] += 1
            return sanitized_pos, warnings
            
        except Exception as e:
            warnings.append(f"Position sanitization failed: {e}, using default")
            return np.zeros(2, dtype=np.float32), warnings
    
    def sanitize_action(self, action: Any) -> Tuple[int, List[str]]:
        """Sanitize agent action."""
        warnings = []
        
        try:
            # Convert to integer
            if isinstance(action, (float, np.floating)):
                action_int = int(round(action))
                if action_int != action:
                    warnings.append(f"Action {action} rounded to {action_int}")
            elif isinstance(action, (int, np.integer)):
                action_int = int(action)
            else:
                warnings.append(f"Invalid action type {type(action)}, using no-op")
                return 0, warnings
            
            # Validate range
            sanitized_action, action_warnings = self.validate_input('action', action_int)
            warnings.extend(action_warnings)
            
            # Clamp to valid range
            if not (0 <= sanitized_action <= 5):
                warnings.append(f"Action {sanitized_action} clamped to valid range")
                sanitized_action = max(0, min(5, sanitized_action))
            
            self.sanitization_stats['sanitized_inputs'] += 1
            return sanitized_action, warnings
            
        except Exception as e:
            warnings.append(f"Action sanitization failed: {e}, using no-op")
            return 0, warnings
    
    def validate_input(self, category: str, value: Any) -> Tuple[Any, List[str]]:
        """Validate input against category rules."""
        self.sanitization_stats['total_checks'] += 1
        warnings = []
        
        if category not in self.rules:
            return value, warnings
        
        for rule in self.rules[category]:
            try:
                if not rule.validator(value):
                    if rule.severity == "error":
                        self.sanitization_stats['violations_detected'] += 1
                        raise ValueError(f"{rule.name}: {rule.error_message}")
                    elif rule.severity == "warning":
                        warnings.append(f"{rule.name}: {rule.error_message}")
                        self.sanitization_stats['violations_detected'] += 1
            except Exception as e:
                if rule.severity == "error":
                    self.sanitization_stats['violations_detected'] += 1
                    raise ValueError(f"{rule.name} validation failed: {e}")
                else:
                    warnings.append(f"{rule.name} validation failed: {e}")
        
        return value, warnings
    
    def _validate_position_coordinates(self, position: Any) -> bool:
        """Validate position coordinates."""
        try:
            if isinstance(position, (list, tuple)):
                pos_array = np.array(position)
            elif isinstance(position, np.ndarray):
                pos_array = position
            else:
                return False
            
            return (len(pos_array) == 2 and 
                   np.all(np.isfinite(pos_array)) and
                   pos_array.dtype in [np.float32, np.float64, int, float])
        except:
            return False
    
    def _validate_position_bounds(self, position: Any) -> bool:
        """Validate position is within reasonable bounds."""
        try:
            pos_array = np.array(position)
            max_reasonable = 1000000  # 1M units
            return np.all(np.abs(pos_array) <= max_reasonable)
        except:
            return False
    
    def _validate_no_script_injection(self, text: str) -> bool:
        """Check for script injection patterns."""
        if not isinstance(text, str):
            return False
        
        # Common injection patterns
        dangerous_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
            r'import\s+',
            r'__import__',
            r'subprocess',
            r'os\.system',
            r'shell=True'
        ]
        
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False
        
        return True
    
    def _validate_no_dangerous_paths(self, config: Any) -> bool:
        """Check for dangerous file paths in configuration."""
        if isinstance(config, dict):
            config_str = str(config)
        else:
            config_str = str(config)
        
        dangerous_paths = [
            '../',
            '/etc/',
            '/proc/',
            '/sys/',
            '~/',
            'C:\\Windows',
            'C:\\System32'
        ]
        
        config_lower = config_str.lower()
        for path in dangerous_paths:
            if path.lower() in config_lower:
                return False
        
        return True
    
    def _sanitize_key_name(self, key: str) -> str:
        """Sanitize configuration key name."""
        # Remove potentially dangerous characters
        safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', str(key))
        
        # Ensure doesn't start with double underscore (Python special methods)
        if safe_key.startswith('__'):
            safe_key = 'safe_' + safe_key
        
        return safe_key[:50]  # Limit length
    
    def _get_safe_default(self, original_value: Any) -> Any:
        """Get safe default value for failed validation."""
        if isinstance(original_value, (int, float)):
            return 0
        elif isinstance(original_value, str):
            return ""
        elif isinstance(original_value, (list, tuple)):
            return []
        elif isinstance(original_value, dict):
            return {}
        elif isinstance(original_value, bool):
            return False
        else:
            return None
    
    def get_sanitization_stats(self) -> Dict[str, Any]:
        """Get sanitization statistics."""
        stats = self.sanitization_stats.copy()
        if stats['total_checks'] > 0:
            stats['violation_rate'] = stats['violations_detected'] / stats['total_checks']
            stats['sanitization_rate'] = stats['sanitized_inputs'] / stats['total_checks']
        else:
            stats['violation_rate'] = 0.0
            stats['sanitization_rate'] = 0.0
        
        return stats


# Global sanitizer instance
_global_sanitizer = InputSanitizer()


def sanitize_agent_input(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Convenience function for sanitizing agent configuration."""
    return _global_sanitizer.sanitize_agent_config(config)


def sanitize_position_input(position: Any) -> Tuple[np.ndarray, List[str]]:
    """Convenience function for sanitizing position input."""
    return _global_sanitizer.sanitize_position(position)


def sanitize_action_input(action: Any) -> Tuple[int, List[str]]:
    """Convenience function for sanitizing action input."""
    return _global_sanitizer.sanitize_action(action)


def get_sanitization_stats() -> Dict[str, Any]:
    """Get global sanitization statistics."""
    return _global_sanitizer.get_sanitization_stats()