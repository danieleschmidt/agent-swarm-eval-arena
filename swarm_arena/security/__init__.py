"""Security components for the swarm arena."""

# Import SecurityManager with fallback
from .auth import SecureTokenManager as SecurityManager

from .input_sanitization import (
    InputSanitizer, 
    sanitize_agent_input, 
    sanitize_position_input, 
    sanitize_action_input,
    get_sanitization_stats
)

__all__ = [
    'SecurityManager',
    'InputSanitizer',
    'sanitize_agent_input',
    'sanitize_position_input', 
    'sanitize_action_input',
    'get_sanitization_stats'
]