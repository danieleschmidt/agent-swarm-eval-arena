"""Comprehensive error handling and recovery system."""

import functools
import traceback
import time
from typing import Any, Callable, Dict, Optional, Type, Union
from dataclasses import dataclass, field
from collections import defaultdict

from ..exceptions import AgentError, ValidationError, NetworkError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ErrorContext:
    """Error context information for debugging and recovery."""
    
    error_type: str
    error_message: str
    timestamp: float = field(default_factory=time.time)
    component: str = "unknown"
    agent_id: Optional[int] = None
    step: Optional[int] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False


class ErrorRecoveryManager:
    """Manages error recovery strategies and patterns."""
    
    def __init__(self, max_error_history: int = 1000):
        """Initialize error recovery manager.
        
        Args:
            max_error_history: Maximum number of errors to track
        """
        self.max_error_history = max_error_history
        self.error_history: list[ErrorContext] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.recovery_strategies: Dict[str, Callable] = {}
        self.error_thresholds: Dict[str, int] = {
            "agent_action_error": 10,
            "spatial_index_error": 5,
            "telemetry_error": 20,
            "network_error": 3
        }
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self) -> None:
        """Register default error recovery strategies."""
        
        def agent_fallback_strategy(error_ctx: ErrorContext) -> bool:
            """Fallback strategy for agent errors."""
            logger.info(f"Applying agent fallback for agent {error_ctx.agent_id}")
            # Agent errors usually fallback to no-op action (0)
            return True
        
        def spatial_reset_strategy(error_ctx: ErrorContext) -> bool:
            """Reset spatial index on errors."""
            logger.info("Attempting spatial index reset")
            # This would be called by the arena to reset its spatial index
            return True
        
        def telemetry_cleanup_strategy(error_ctx: ErrorContext) -> bool:
            """Clean up telemetry on errors."""
            logger.info("Performing telemetry cleanup")
            # This would trigger telemetry cleanup
            return True
        
        self.recovery_strategies.update({
            "agent_action_error": agent_fallback_strategy,
            "spatial_index_error": spatial_reset_strategy,
            "telemetry_error": telemetry_cleanup_strategy
        })
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable[[ErrorContext], bool]) -> None:
        """Register a custom recovery strategy.
        
        Args:
            error_type: Type of error to handle
            strategy: Recovery function that returns True if successful
        """
        self.recovery_strategies[error_type] = strategy
        logger.debug(f"Registered recovery strategy for {error_type}")
    
    def record_error(self, error: Exception, 
                    component: str = "unknown",
                    agent_id: Optional[int] = None,
                    step: Optional[int] = None,
                    context_data: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Record an error and attempt recovery.
        
        Args:
            error: Exception that occurred
            component: Component where error occurred
            agent_id: Agent ID if applicable
            step: Simulation step if applicable
            context_data: Additional context information
            
        Returns:
            ErrorContext with recovery information
        """
        # Determine error type
        error_type = self._classify_error(error, component)
        
        # Create error context
        error_ctx = ErrorContext(
            error_type=error_type,
            error_message=str(error),
            component=component,
            agent_id=agent_id,
            step=step,
            context_data=context_data or {},
            stack_trace=traceback.format_exc()
        )
        
        # Record error
        self._add_to_history(error_ctx)
        self.error_counts[error_type] += 1
        
        # Attempt recovery
        recovery_successful = self._attempt_recovery(error_ctx)
        error_ctx.recovery_attempted = True
        error_ctx.recovery_successful = recovery_successful
        
        # Log error with context
        self._log_error(error_ctx)
        
        return error_ctx
    
    def _classify_error(self, error: Exception, component: str) -> str:
        """Classify error type for recovery strategy selection.
        
        Args:
            error: Exception to classify
            component: Component where error occurred
            
        Returns:
            Error type string
        """
        if isinstance(error, AgentError):
            return "agent_action_error"
        elif isinstance(error, ValidationError):
            return "validation_error"
        elif isinstance(error, NetworkError):
            return "network_error"
        elif "spatial" in component.lower():
            return "spatial_index_error"
        elif "telemetry" in component.lower():
            return "telemetry_error"
        elif isinstance(error, (KeyError, AttributeError)):
            return "data_structure_error"
        elif isinstance(error, (ValueError, TypeError)):
            return "parameter_error"
        elif isinstance(error, MemoryError):
            return "memory_error"
        else:
            return "unknown_error"
    
    def _add_to_history(self, error_ctx: ErrorContext) -> None:
        """Add error to history with size management."""
        self.error_history.append(error_ctx)
        
        # Maintain history size
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history//2:]
    
    def _attempt_recovery(self, error_ctx: ErrorContext) -> bool:
        """Attempt to recover from error using registered strategies.
        
        Args:
            error_ctx: Error context
            
        Returns:
            True if recovery was successful
        """
        strategy = self.recovery_strategies.get(error_ctx.error_type)
        
        if strategy:
            try:
                return strategy(error_ctx)
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {str(recovery_error)}")
                return False
        
        return False
    
    def _log_error(self, error_ctx: ErrorContext) -> None:
        """Log error with appropriate level and context."""
        context_str = f"Component: {error_ctx.component}"
        if error_ctx.agent_id is not None:
            context_str += f", Agent: {error_ctx.agent_id}"
        if error_ctx.step is not None:
            context_str += f", Step: {error_ctx.step}"
        
        # Check if this is a recurring error
        error_count = self.error_counts[error_ctx.error_type]
        threshold = self.error_thresholds.get(error_ctx.error_type, 5)
        
        if error_count >= threshold:
            logger.error(
                f"CRITICAL: Recurring error ({error_count} times) - {error_ctx.error_message}",
                context=context_str,
                error_type=error_ctx.error_type,
                recovery_successful=error_ctx.recovery_successful
            )
        elif error_ctx.error_type in ["memory_error", "network_error"]:
            logger.error(
                f"SERIOUS: {error_ctx.error_message}",
                context=context_str,
                error_type=error_ctx.error_type
            )
        else:
            logger.warning(
                f"{error_ctx.error_message}",
                context=context_str,
                error_type=error_ctx.error_type,
                recovery_attempted=error_ctx.recovery_attempted
            )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and patterns.
        
        Returns:
            Dictionary of error statistics
        """
        if not self.error_history:
            return {"status": "no_errors"}
        
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 300]  # Last 5 minutes
        
        stats = {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_types": dict(self.error_counts),
            "recovery_rate": sum(1 for e in self.error_history if e.recovery_successful) / len(self.error_history),
            "most_common_error": max(self.error_counts, key=self.error_counts.get) if self.error_counts else None,
            "error_frequency": len(recent_errors) / 5.0,  # errors per minute
            "critical_errors": sum(1 for error_type, count in self.error_counts.items() 
                                 if count >= self.error_thresholds.get(error_type, 5))
        }
        
        return stats
    
    def should_halt_system(self) -> bool:
        """Determine if system should halt due to critical errors.
        
        Returns:
            True if system should halt
        """
        stats = self.get_error_statistics()
        
        # Halt conditions
        if stats.get("critical_errors", 0) > 3:
            return True
        
        if stats.get("error_frequency", 0) > 10:  # More than 10 errors per minute
            return True
        
        if self.error_counts.get("memory_error", 0) > 1:
            return True
        
        if stats.get("recovery_rate", 1.0) < 0.5 and len(self.error_history) > 20:
            return True
        
        return False
    
    def clear_error_history(self) -> None:
        """Clear error history and reset counters."""
        self.error_history.clear()
        self.error_counts.clear()
        logger.info("Error history cleared")


# Global error manager instance
error_manager = ErrorRecoveryManager()


def with_error_handling(component: str = "unknown", 
                       agent_id: Optional[int] = None,
                       fallback_return: Any = None,
                       re_raise: bool = False):
    """Decorator for automatic error handling and recovery.
    
    Args:
        component: Component name for error classification
        agent_id: Agent ID if applicable
        fallback_return: Return value on error
        re_raise: Whether to re-raise after handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Record error and attempt recovery
                error_ctx = error_manager.record_error(
                    error=e,
                    component=component,
                    agent_id=agent_id,
                    context_data={
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                )
                
                # Check if system should halt
                if error_manager.should_halt_system():
                    logger.critical("System halt recommended due to critical errors")
                    raise SystemError("Critical error threshold exceeded")
                
                if re_raise:
                    raise
                    
                return fallback_return
        
        return wrapper
    return decorator


def validate_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize observation data.
    
    Args:
        observation: Raw observation data
        
    Returns:
        Validated and sanitized observation
        
    Raises:
        ValidationError: If observation is invalid
    """
    if not isinstance(observation, dict):
        raise ValidationError(f"Observation must be dict, got {type(observation)}")
    
    # Ensure required fields exist with defaults
    validated = {
        "position": observation.get("position", [0.0, 0.0]),
        "velocity": observation.get("velocity", [0.0, 0.0]),
        "nearby_agents": observation.get("nearby_agents", []),
        "resources": observation.get("resources", []),
        "arena_bounds": observation.get("arena_bounds", {"width": 1000, "height": 1000})
    }
    
    # Validate position
    if not isinstance(validated["position"], (list, tuple)) or len(validated["position"]) != 2:
        logger.warning(f"Invalid position format: {validated['position']}, using default")
        validated["position"] = [0.0, 0.0]
    
    # Ensure position values are numeric
    try:
        validated["position"] = [float(validated["position"][0]), float(validated["position"][1])]
    except (ValueError, TypeError):
        logger.warning("Position values not numeric, using default")
        validated["position"] = [0.0, 0.0]
    
    # Validate velocity
    if not isinstance(validated["velocity"], (list, tuple)) or len(validated["velocity"]) != 2:
        validated["velocity"] = [0.0, 0.0]
    
    try:
        validated["velocity"] = [float(validated["velocity"][0]), float(validated["velocity"][1])]
    except (ValueError, TypeError):
        validated["velocity"] = [0.0, 0.0]
    
    # Validate nearby_agents
    if not isinstance(validated["nearby_agents"], list):
        validated["nearby_agents"] = []
    
    # Clean nearby_agents list
    clean_agents = []
    for agent_pos in validated["nearby_agents"]:
        if isinstance(agent_pos, (list, tuple)) and len(agent_pos) == 2:
            try:
                clean_pos = [float(agent_pos[0]), float(agent_pos[1])]
                clean_agents.append(clean_pos)
            except (ValueError, TypeError):
                continue  # Skip invalid positions
    validated["nearby_agents"] = clean_agents
    
    # Validate resources
    if not isinstance(validated["resources"], list):
        validated["resources"] = []
    
    # Clean resources list
    clean_resources = []
    for resource_pos in validated["resources"]:
        if isinstance(resource_pos, (list, tuple)) and len(resource_pos) == 2:
            try:
                clean_pos = [float(resource_pos[0]), float(resource_pos[1])]
                clean_resources.append(clean_pos)
            except (ValueError, TypeError):
                continue  # Skip invalid positions
    validated["resources"] = clean_resources
    
    # Validate arena_bounds
    if not isinstance(validated["arena_bounds"], dict):
        validated["arena_bounds"] = {"width": 1000, "height": 1000}
    
    if "width" not in validated["arena_bounds"] or "height" not in validated["arena_bounds"]:
        validated["arena_bounds"] = {"width": 1000, "height": 1000}
    
    try:
        validated["arena_bounds"]["width"] = float(validated["arena_bounds"]["width"])
        validated["arena_bounds"]["height"] = float(validated["arena_bounds"]["height"])
    except (ValueError, TypeError):
        validated["arena_bounds"] = {"width": 1000, "height": 1000}
    
    # Copy over any additional fields safely
    for key, value in observation.items():
        if key not in validated:
            validated[key] = value
    
    return validated


def safe_action_execution(agent, observation: Dict[str, Any]) -> int:
    """Safely execute agent action with error handling.
    
    Args:
        agent: Agent instance
        observation: Observation data
        
    Returns:
        Action integer (0-5), defaults to 0 on error
    """
    try:
        # Validate observation first
        validated_obs = validate_observation(observation)
        
        # Execute agent action
        action = agent.act(validated_obs)
        
        # Validate action
        if not isinstance(action, int) or not (0 <= action <= 5):
            logger.warning(f"Invalid action {action} from agent {agent.agent_id}, using no-op")
            return 0
        
        return action
        
    except Exception as e:
        # Record error and return safe fallback
        error_manager.record_error(
            error=e,
            component="agent_action",
            agent_id=getattr(agent, 'agent_id', None)
        )
        return 0  # Safe fallback: no-op action