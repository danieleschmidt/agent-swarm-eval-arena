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