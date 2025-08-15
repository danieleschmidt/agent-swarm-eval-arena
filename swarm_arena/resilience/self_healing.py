"""
Self-Healing Infrastructure for Autonomous SDLC
Implements adaptive systems that learn and evolve from failures.
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import numpy as np
from pathlib import Path

from ..exceptions import SystemError, CircuitBreakerError
from ..utils.logging import get_logger


class HealingStrategy(Enum):
    """Available self-healing strategies."""
    RESTART = "restart"
    FALLBACK = "fallback"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CIRCUIT_BREAK = "circuit_break"
    PARAMETER_ADJUST = "parameter_adjust"
    ROUTE_AROUND = "route_around"


@dataclass
class FailurePattern:
    """Pattern recognition for common failure modes."""
    pattern_id: str
    description: str
    indicators: List[str]
    healing_strategy: HealingStrategy
    confidence: float = 0.0
    occurrence_count: int = 0
    last_seen: float = field(default_factory=time.time)
    success_rate: float = 0.0


@dataclass
class HealingAction:
    """Self-healing action to be executed."""
    action_id: str
    strategy: HealingStrategy
    target_component: str
    parameters: Dict[str, Any]
    priority: int = 0  # 0 = highest priority
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    executed_at: Optional[float] = None
    completed_at: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None


class AdaptiveMemory:
    """Memory system that learns from past healing actions."""
    
    def __init__(self, memory_size: int = 10000):
        self.memory_size = memory_size
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self.action_history: deque = deque(maxlen=memory_size)
        self.success_rates: Dict[HealingStrategy, float] = defaultdict(float)
        self.pattern_lock = threading.Lock()
        self.logger = get_logger(__name__)
    
    def record_failure(self, component: str, error: Exception, context: Dict[str, Any]) -> str:
        """Record a failure and attempt to match it to known patterns."""
        failure_signature = self._create_failure_signature(component, error, context)
        
        with self.pattern_lock:
            pattern_id = self._match_or_create_pattern(failure_signature, error, context)
            self.failure_patterns[pattern_id].occurrence_count += 1
            self.failure_patterns[pattern_id].last_seen = time.time()
        
        self.logger.info(f"Recorded failure pattern: {pattern_id} for component: {component}")
        return pattern_id
    
    def record_healing_action(self, action: HealingAction) -> None:
        """Record the result of a healing action."""
        self.action_history.append(action)
        
        if action.success is not None:
            # Update strategy success rate
            strategy_actions = [a for a in self.action_history if a.strategy == action.strategy]
            successful_actions = [a for a in strategy_actions if a.success]
            
            if strategy_actions:
                self.success_rates[action.strategy] = len(successful_actions) / len(strategy_actions)
        
        self.logger.info(f"Recorded healing action: {action.action_id}, success: {action.success}")
    
    def recommend_healing_strategy(self, pattern_id: str, component: str) -> HealingStrategy:
        """Recommend the best healing strategy based on historical success."""
        if pattern_id in self.failure_patterns:
            pattern = self.failure_patterns[pattern_id]
            
            # If pattern has high confidence and good success rate, use its strategy
            if pattern.confidence > 0.8 and pattern.success_rate > 0.5:
                return pattern.healing_strategy
        
        # Otherwise, recommend based on overall strategy success rates
        if self.success_rates:
            best_strategy = max(self.success_rates.items(), key=lambda x: x[1])[0]
            return best_strategy
        
        # Default fallback
        return HealingStrategy.RESTART
    
    def _create_failure_signature(self, component: str, error: Exception, context: Dict[str, Any]) -> str:
        """Create a unique signature for this type of failure."""
        error_type = type(error).__name__
        error_message = str(error)[:100]  # First 100 chars
        
        # Include relevant context
        context_signature = []
        for key in ['load', 'memory_usage', 'cpu_usage', 'agent_count']:
            if key in context:
                context_signature.append(f"{key}:{context[key]}")
        
        signature = f"{component}:{error_type}:{error_message}:{':'.join(context_signature)}"
        return signature
    
    def _match_or_create_pattern(self, signature: str, error: Exception, context: Dict[str, Any]) -> str:
        """Match failure to existing pattern or create new one."""
        # Simple matching based on component and error type
        component = signature.split(':')[0]
        error_type = signature.split(':')[1]
        pattern_key = f"{component}_{error_type}"
        
        if pattern_key not in self.failure_patterns:
            # Create new pattern
            strategy = self._infer_initial_strategy(error, context)
            self.failure_patterns[pattern_key] = FailurePattern(
                pattern_id=pattern_key,
                description=f"{error_type} in {component}",
                indicators=[signature],
                healing_strategy=strategy,
                confidence=0.1  # Start with low confidence
            )
        else:
            # Update existing pattern
            pattern = self.failure_patterns[pattern_key]
            pattern.confidence = min(1.0, pattern.confidence + 0.1)
            if signature not in pattern.indicators:
                pattern.indicators.append(signature)
        
        return pattern_key
    
    def _infer_initial_strategy(self, error: Exception, context: Dict[str, Any]) -> HealingStrategy:
        """Infer initial healing strategy based on error type and context."""
        error_type = type(error).__name__
        
        # Memory-related errors
        if 'memory' in error_type.lower() or 'oom' in str(error).lower():
            return HealingStrategy.SCALE_DOWN
        
        # Connection/network errors
        if 'connection' in error_type.lower() or 'network' in error_type.lower():
            return HealingStrategy.ROUTE_AROUND
        
        # Resource exhaustion
        if 'resource' in str(error).lower():
            return HealingStrategy.SCALE_UP
        
        # Configuration errors
        if 'config' in error_type.lower() or 'validation' in error_type.lower():
            return HealingStrategy.PARAMETER_ADJUST
        
        # Default to restart for unknown errors
        return HealingStrategy.RESTART


class SelfHealingSystem:
    """Autonomous self-healing system that adapts and learns."""
    
    def __init__(self, enable_learning: bool = True):
        self.enable_learning = enable_learning
        self.memory = AdaptiveMemory()
        self.healing_actions: Dict[str, HealingAction] = {}
        self.component_health: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.healing_lock = threading.Lock()
        self.running = False
        self.monitor_thread = None
        self.logger = get_logger(__name__)
        
        # Performance metrics
        self.metrics = {
            'total_failures': 0,
            'successful_healings': 0,
            'failed_healings': 0,
            'automatic_recoveries': 0,
            'mttr': 0.0,  # Mean Time to Recovery
            'mtbf': 0.0   # Mean Time Between Failures
        }
        
        # Register default healing functions
        self._register_healing_functions()
    
    def start(self) -> None:
        """Start the self-healing monitoring system."""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Self-healing system started")
    
    def stop(self) -> None:
        """Stop the self-healing system."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Self-healing system stopped")
    
    def report_failure(
        self, 
        component: str, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None,
        auto_heal: bool = True
    ) -> str:
        """Report a failure and optionally trigger automatic healing."""
        if context is None:
            context = {}
        
        self.metrics['total_failures'] += 1
        
        # Record in adaptive memory
        pattern_id = self.memory.record_failure(component, error, context)
        
        # Update component health
        self.component_health[component].update({
            'status': 'failed',
            'last_error': str(error),
            'last_failure_time': time.time(),
            'pattern_id': pattern_id
        })
        
        if auto_heal:
            healing_action = self._create_healing_action(component, pattern_id, context)
            self._execute_healing_action(healing_action)
            return healing_action.action_id
        
        return pattern_id
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status of a component."""
        return self.component_health.get(component, {'status': 'unknown'})
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        total_healings = self.metrics['successful_healings'] + self.metrics['failed_healings']
        
        return {
            **self.metrics,
            'healing_success_rate': (
                self.metrics['successful_healings'] / total_healings 
                if total_healings > 0 else 0.0
            ),
            'active_components': len(self.component_health),
            'memory_patterns': len(self.memory.failure_patterns),
            'strategy_success_rates': dict(self.memory.success_rates)
        }
    
    def _create_healing_action(
        self, 
        component: str, 
        pattern_id: str, 
        context: Dict[str, Any]
    ) -> HealingAction:
        """Create a healing action based on the failure pattern."""
        strategy = self.memory.recommend_healing_strategy(pattern_id, component)
        
        action = HealingAction(
            action_id=f"heal_{component}_{int(time.time())}",
            strategy=strategy,
            target_component=component,
            parameters=self._get_strategy_parameters(strategy, component, context),
            priority=self._calculate_priority(component, pattern_id)
        )
        
        return action
    
    def _execute_healing_action(self, action: HealingAction) -> None:
        """Execute a healing action."""
        with self.healing_lock:
            self.healing_actions[action.action_id] = action
            action.executed_at = time.time()
        
        self.logger.info(f"Executing healing action: {action.action_id} ({action.strategy.value})")
        
        try:
            success = self._apply_healing_strategy(action)
            action.success = success
            action.completed_at = time.time()
            
            if success:
                self.metrics['successful_healings'] += 1
                self._update_component_health(action.target_component, 'healthy')
                self.logger.info(f"Healing action succeeded: {action.action_id}")
            else:
                self.metrics['failed_healings'] += 1
                self.logger.warning(f"Healing action failed: {action.action_id}")
                
                # Try alternative strategy if retries available
                if action.retry_count < action.max_retries:
                    self._retry_with_alternative(action)
        
        except Exception as e:
            action.success = False
            action.error_message = str(e)
            action.completed_at = time.time()
            self.metrics['failed_healings'] += 1
            self.logger.error(f"Healing action error: {action.action_id}: {e}")
        
        finally:
            # Record in memory for learning
            self.memory.record_healing_action(action)
    
    def _apply_healing_strategy(self, action: HealingAction) -> bool:
        """Apply the specific healing strategy."""
        strategy_map = {
            HealingStrategy.RESTART: self._heal_restart,
            HealingStrategy.FALLBACK: self._heal_fallback,
            HealingStrategy.SCALE_UP: self._heal_scale_up,
            HealingStrategy.SCALE_DOWN: self._heal_scale_down,
            HealingStrategy.CIRCUIT_BREAK: self._heal_circuit_break,
            HealingStrategy.PARAMETER_ADJUST: self._heal_parameter_adjust,
            HealingStrategy.ROUTE_AROUND: self._heal_route_around
        }
        
        healing_func = strategy_map.get(action.strategy)
        if healing_func:
            return healing_func(action)
        
        self.logger.error(f"Unknown healing strategy: {action.strategy}")
        return False
    
    def _heal_restart(self, action: HealingAction) -> bool:
        """Restart component healing strategy."""
        component = action.target_component
        self.logger.info(f"Restarting component: {component}")
        
        # Simulate component restart
        time.sleep(0.1)  # Simulate restart time
        
        # Mark as healthy
        self._update_component_health(component, 'healthy')
        return True
    
    def _heal_fallback(self, action: HealingAction) -> bool:
        """Fallback mode healing strategy."""
        component = action.target_component
        fallback_config = action.parameters.get('fallback_config', {})
        
        self.logger.info(f"Switching {component} to fallback mode")
        
        # Apply fallback configuration
        self.component_health[component]['mode'] = 'fallback'
        self.component_health[component]['config'] = fallback_config
        
        return True
    
    def _heal_scale_up(self, action: HealingAction) -> bool:
        """Scale up resources healing strategy."""
        component = action.target_component
        scale_factor = action.parameters.get('scale_factor', 1.5)
        
        self.logger.info(f"Scaling up {component} by factor {scale_factor}")
        
        # Simulate resource scaling
        current_resources = self.component_health[component].get('resources', 1.0)
        new_resources = current_resources * scale_factor
        
        self.component_health[component]['resources'] = new_resources
        self._update_component_health(component, 'healthy')
        
        return True
    
    def _heal_scale_down(self, action: HealingAction) -> bool:
        """Scale down resources healing strategy."""
        component = action.target_component
        scale_factor = action.parameters.get('scale_factor', 0.7)
        
        self.logger.info(f"Scaling down {component} by factor {scale_factor}")
        
        # Simulate resource scaling
        current_resources = self.component_health[component].get('resources', 1.0)
        new_resources = max(0.1, current_resources * scale_factor)  # Minimum threshold
        
        self.component_health[component]['resources'] = new_resources
        self._update_component_health(component, 'healthy')
        
        return True
    
    def _heal_circuit_break(self, action: HealingAction) -> bool:
        """Circuit breaker healing strategy."""
        component = action.target_component
        timeout = action.parameters.get('timeout', 60.0)
        
        self.logger.info(f"Activating circuit breaker for {component}")
        
        # Activate circuit breaker
        self.component_health[component]['circuit_breaker'] = {
            'active': True,
            'timeout': timeout,
            'activated_at': time.time()
        }
        
        return True
    
    def _heal_parameter_adjust(self, action: HealingAction) -> bool:
        """Parameter adjustment healing strategy."""
        component = action.target_component
        adjustments = action.parameters.get('adjustments', {})
        
        self.logger.info(f"Adjusting parameters for {component}: {adjustments}")
        
        # Apply parameter adjustments
        if 'config' not in self.component_health[component]:
            self.component_health[component]['config'] = {}
        
        self.component_health[component]['config'].update(adjustments)
        self._update_component_health(component, 'healthy')
        
        return True
    
    def _heal_route_around(self, action: HealingAction) -> bool:
        """Route around failure healing strategy."""
        component = action.target_component
        alternative_route = action.parameters.get('alternative', f"{component}_backup")
        
        self.logger.info(f"Routing around {component} to {alternative_route}")
        
        # Set up routing bypass
        self.component_health[component]['route_bypass'] = {
            'active': True,
            'alternative': alternative_route,
            'activated_at': time.time()
        }
        
        return True
    
    def _get_strategy_parameters(
        self, 
        strategy: HealingStrategy, 
        component: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get parameters for specific healing strategy."""
        base_params = {}
        
        if strategy == HealingStrategy.SCALE_UP:
            # Scale based on current load
            load = context.get('load', 0.5)
            base_params['scale_factor'] = 1.0 + (load * 0.5)  # Scale more for higher load
        
        elif strategy == HealingStrategy.SCALE_DOWN:
            # Scale down based on memory pressure
            memory_usage = context.get('memory_usage', 0.5)
            base_params['scale_factor'] = max(0.5, 1.0 - (memory_usage * 0.3))
        
        elif strategy == HealingStrategy.PARAMETER_ADJUST:
            # Adjust based on error patterns
            if 'timeout' in str(context.get('error', '')).lower():
                base_params['adjustments'] = {'timeout': 60.0}
            elif 'batch' in str(context.get('error', '')).lower():
                base_params['adjustments'] = {'batch_size': 16}
        
        elif strategy == HealingStrategy.CIRCUIT_BREAK:
            # Longer timeout for critical components
            base_params['timeout'] = 120.0 if 'critical' in component else 60.0
        
        return base_params
    
    def _calculate_priority(self, component: str, pattern_id: str) -> int:
        """Calculate healing action priority."""
        # Higher priority (lower number) for critical components
        if 'critical' in component or 'core' in component:
            return 0
        elif 'important' in component:
            return 1
        else:
            return 2
    
    def _update_component_health(self, component: str, status: str) -> None:
        """Update component health status."""
        self.component_health[component]['status'] = status
        self.component_health[component]['last_update'] = time.time()
        
        if status == 'healthy':
            # Clear error information
            self.component_health[component].pop('last_error', None)
            self.component_health[component].pop('pattern_id', None)
    
    def _retry_with_alternative(self, failed_action: HealingAction) -> None:
        """Retry with alternative strategy."""
        failed_action.retry_count += 1
        
        # Try different strategy
        alternative_strategies = [s for s in HealingStrategy if s != failed_action.strategy]
        if alternative_strategies:
            alt_strategy = alternative_strategies[0]  # Simple selection
            
            retry_action = HealingAction(
                action_id=f"{failed_action.action_id}_retry_{failed_action.retry_count}",
                strategy=alt_strategy,
                target_component=failed_action.target_component,
                parameters=self._get_strategy_parameters(
                    alt_strategy, 
                    failed_action.target_component, 
                    {}
                ),
                retry_count=failed_action.retry_count
            )
            
            self.logger.info(f"Retrying with alternative strategy: {alt_strategy.value}")
            self._execute_healing_action(retry_action)
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop for proactive healing."""
        while self.running:
            try:
                self._proactive_health_check()
                time.sleep(10.0)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                time.sleep(1.0)
    
    def _proactive_health_check(self) -> None:
        """Proactive health monitoring and prevention."""
        current_time = time.time()
        
        for component, health in self.component_health.items():
            # Check for circuit breaker timeout
            if 'circuit_breaker' in health and health['circuit_breaker']['active']:
                cb = health['circuit_breaker']
                if current_time - cb['activated_at'] > cb['timeout']:
                    # Reset circuit breaker
                    health['circuit_breaker']['active'] = False
                    self._update_component_health(component, 'healthy')
                    self.logger.info(f"Circuit breaker reset for {component}")
            
            # Check for route bypass cleanup
            if 'route_bypass' in health and health['route_bypass']['active']:
                bypass = health['route_bypass']
                if current_time - bypass['activated_at'] > 300:  # 5 minutes
                    # Try to restore normal routing
                    if self._test_component_health(component):
                        health['route_bypass']['active'] = False
                        self._update_component_health(component, 'healthy')
                        self.logger.info(f"Route bypass cleared for {component}")
    
    def _test_component_health(self, component: str) -> bool:
        """Test if component is healthy."""
        # Simplified health test
        return True  # Assume healthy for demo
    
    def _register_healing_functions(self) -> None:
        """Register additional healing functions."""
        # This can be extended to register custom healing strategies
        pass
    
    def export_learning_data(self, filepath: str) -> None:
        """Export learned patterns and strategies for analysis."""
        export_data = {
            'failure_patterns': {
                pid: {
                    'pattern_id': pattern.pattern_id,
                    'description': pattern.description,
                    'healing_strategy': pattern.healing_strategy.value,
                    'confidence': pattern.confidence,
                    'occurrence_count': pattern.occurrence_count,
                    'success_rate': pattern.success_rate
                }
                for pid, pattern in self.memory.failure_patterns.items()
            },
            'strategy_success_rates': {
                strategy.value: rate 
                for strategy, rate in self.memory.success_rates.items()
            },
            'system_metrics': self.get_system_metrics(),
            'exported_at': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Learning data exported to {filepath}")


# Global self-healing system instance
_global_healing_system: Optional[SelfHealingSystem] = None


def get_healing_system() -> SelfHealingSystem:
    """Get or create global self-healing system."""
    global _global_healing_system
    if _global_healing_system is None:
        _global_healing_system = SelfHealingSystem()
        _global_healing_system.start()
    return _global_healing_system


def heal_on_failure(component: str):
    """Decorator to automatically heal on function failures."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                healing_system = get_healing_system()
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
                healing_system.report_failure(component, e, context)
                raise  # Re-raise after reporting
        return wrapper
    return decorator