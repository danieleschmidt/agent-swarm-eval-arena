"""Custom exceptions for the Swarm Arena."""


class SwarmArenaError(Exception):
    """Base exception for all Swarm Arena errors."""
    pass


class ConfigurationError(SwarmArenaError):
    """Raised when there's an error in configuration."""
    pass


class AgentError(SwarmArenaError):
    """Raised when there's an error with agent operations."""
    pass


class EnvironmentError(SwarmArenaError):
    """Raised when there's an error with environment operations."""
    pass


class PhysicsError(SwarmArenaError):
    """Raised when there's an error in physics calculations."""
    pass


class ValidationError(SwarmArenaError):
    """Raised when validation fails."""
    pass


class ResourceError(SwarmArenaError):
    """Raised when there's an error with resource management."""
    pass


class SimulationError(SwarmArenaError):
    """Raised when there's an error during simulation."""
    pass


class NetworkError(SwarmArenaError):
    """Raised when there's a network-related error."""
    pass


class RayError(SwarmArenaError):
    """Raised when there's an error with Ray distributed computing."""
    pass


class CircuitBreakerError(SwarmArenaError):
    """Raised when circuit breaker is open."""
    pass


class SystemError(SwarmArenaError):
    """Raised when there's a system-level error."""
    pass


class SecurityError(SwarmArenaError):
    """Raised when there's a security-related error."""
    pass


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""
    pass