"""Auto-scaling engine for dynamic resource management."""

import time
import threading
import math
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import psutil

from ..utils.logging import get_logger
from ..resilience.circuit_breaker import circuit_breaker, CONFIGS

logger = get_logger(__name__)


class ScalingDirection(Enum):
    """Scaling direction enum."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    agent_count: int = 0
    steps_per_second: float = 0.0
    queue_length: int = 0
    response_time_p95: float = 0.0
    active_connections: int = 0


@dataclass
class ScalingPolicy:
    """Scaling policy configuration."""
    name: str
    
    # Scale up thresholds
    cpu_scale_up_threshold: float = 0.75
    memory_scale_up_threshold: float = 0.80
    queue_scale_up_threshold: int = 100
    response_time_scale_up_threshold: float = 5.0
    
    # Scale down thresholds
    cpu_scale_down_threshold: float = 0.30
    memory_scale_down_threshold: float = 0.40
    queue_scale_down_threshold: int = 10
    response_time_scale_down_threshold: float = 1.0
    
    # Scaling parameters
    min_instances: int = 1
    max_instances: int = 10
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    cooldown_period: float = 300.0  # 5 minutes
    evaluation_period: float = 60.0  # 1 minute
    
    # Stability requirements
    consecutive_periods_required: int = 3


class ScalingDecision:
    """Represents a scaling decision."""
    
    def __init__(self, direction: ScalingDirection, target_instances: int, 
                 reason: str, confidence: float = 1.0):
        self.direction = direction
        self.target_instances = target_instances
        self.reason = reason
        self.confidence = confidence
        self.timestamp = time.time()


class MetricsCollector:
    """Collects metrics for scaling decisions."""
    
    def __init__(self):
        self.agent_count = 0
        self.queue_length = 0
        self.active_connections = 0
        self.step_times = []
        self.response_times = []
        self._lock = threading.Lock()
    
    def update_agent_count(self, count: int) -> None:
        """Update current agent count."""
        with self._lock:
            self.agent_count = count
    
    def update_queue_length(self, length: int) -> None:
        """Update current queue length."""
        with self._lock:
            self.queue_length = length
    
    def update_active_connections(self, count: int) -> None:
        """Update active connection count."""
        with self._lock:
            self.active_connections = count
    
    def record_step_time(self, duration: float) -> None:
        """Record simulation step time."""
        with self._lock:
            self.step_times.append(duration)
            # Keep only recent measurements
            if len(self.step_times) > 1000:
                self.step_times = self.step_times[-500:]
    
    def record_response_time(self, duration: float) -> None:
        """Record response time."""
        with self._lock:
            self.response_times.append(duration)
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-500:]
    
    def get_current_metrics(self) -> ScalingMetrics:
        """Get current scaling metrics."""
        with self._lock:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
            memory_usage = psutil.virtual_memory().percent / 100.0
            
            # Performance metrics
            steps_per_second = 0.0
            if self.step_times:
                avg_step_time = sum(self.step_times[-100:]) / len(self.step_times[-100:])
                if avg_step_time > 0:
                    steps_per_second = 1.0 / avg_step_time
            
            # Response time P95
            response_time_p95 = 0.0
            if self.response_times:
                sorted_times = sorted(self.response_times[-100:])
                p95_index = int(0.95 * len(sorted_times))
                if p95_index < len(sorted_times):
                    response_time_p95 = sorted_times[p95_index]
            
            return ScalingMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                agent_count=self.agent_count,
                steps_per_second=steps_per_second,
                queue_length=self.queue_length,
                response_time_p95=response_time_p95,
                active_connections=self.active_connections
            )


class ScalingEngine:
    """Core auto-scaling engine."""
    
    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.metrics_collector = MetricsCollector()
        self.current_instances = policy.min_instances
        self.last_scaling_time = 0.0
        self.scaling_history = []
        self.consecutive_decisions = []
        self._lock = threading.Lock()
        
        logger.info(f"Scaling engine initialized with policy: {policy.name}")
    
    def evaluate_scaling(self) -> Optional[ScalingDecision]:
        """Evaluate if scaling is needed."""
        with self._lock:
            metrics = self.metrics_collector.get_current_metrics()
            
            # Check cooldown period
            if time.time() - self.last_scaling_time < self.policy.cooldown_period:
                return None
            
            # Evaluate scaling need
            decision = self._make_scaling_decision(metrics)
            
            if decision.direction != ScalingDirection.NONE:
                # Track consecutive decisions
                self.consecutive_decisions.append(decision)
                
                # Only act if we have enough consecutive decisions
                if len(self.consecutive_decisions) >= self.policy.consecutive_periods_required:
                    # Check if all recent decisions agree
                    recent_directions = [d.direction for d in self.consecutive_decisions[-self.policy.consecutive_periods_required:]]
                    if len(set(recent_directions)) == 1:  # All decisions agree
                        return decision
            
            return None
    
    def _make_scaling_decision(self, metrics: ScalingMetrics) -> ScalingDecision:
        """Make scaling decision based on metrics."""
        scale_up_score = self._calculate_scale_up_score(metrics)
        scale_down_score = self._calculate_scale_down_score(metrics)
        
        if scale_up_score > 0.7 and self.current_instances < self.policy.max_instances:
            target_instances = min(
                self.policy.max_instances,
                math.ceil(self.current_instances * self.policy.scale_up_factor)
            )
            
            reasons = []
            if metrics.cpu_usage > self.policy.cpu_scale_up_threshold:
                reasons.append(f"High CPU: {metrics.cpu_usage:.1%}")
            if metrics.memory_usage > self.policy.memory_scale_up_threshold:
                reasons.append(f"High memory: {metrics.memory_usage:.1%}")
            if metrics.queue_length > self.policy.queue_scale_up_threshold:
                reasons.append(f"Queue length: {metrics.queue_length}")
            if metrics.response_time_p95 > self.policy.response_time_scale_up_threshold:
                reasons.append(f"High response time: {metrics.response_time_p95:.2f}s")
            
            return ScalingDecision(
                ScalingDirection.UP,
                target_instances,
                f"Scale up needed: {', '.join(reasons)}",
                scale_up_score
            )
        
        elif scale_down_score > 0.7 and self.current_instances > self.policy.min_instances:
            target_instances = max(
                self.policy.min_instances,
                math.floor(self.current_instances * self.policy.scale_down_factor)
            )
            
            reasons = []
            if metrics.cpu_usage < self.policy.cpu_scale_down_threshold:
                reasons.append(f"Low CPU: {metrics.cpu_usage:.1%}")
            if metrics.memory_usage < self.policy.memory_scale_down_threshold:
                reasons.append(f"Low memory: {metrics.memory_usage:.1%}")
            if metrics.queue_length < self.policy.queue_scale_down_threshold:
                reasons.append(f"Low queue: {metrics.queue_length}")
            
            return ScalingDecision(
                ScalingDirection.DOWN,
                target_instances,
                f"Scale down possible: {', '.join(reasons)}",
                scale_down_score
            )
        
        return ScalingDecision(ScalingDirection.NONE, self.current_instances, "No scaling needed")
    
    def _calculate_scale_up_score(self, metrics: ScalingMetrics) -> float:
        """Calculate scale up score (0-1)."""
        scores = []
        
        # CPU score
        if metrics.cpu_usage > self.policy.cpu_scale_up_threshold:
            scores.append(min(1.0, metrics.cpu_usage / self.policy.cpu_scale_up_threshold))
        
        # Memory score
        if metrics.memory_usage > self.policy.memory_scale_up_threshold:
            scores.append(min(1.0, metrics.memory_usage / self.policy.memory_scale_up_threshold))
        
        # Queue score
        if metrics.queue_length > self.policy.queue_scale_up_threshold:
            scores.append(min(1.0, metrics.queue_length / self.policy.queue_scale_up_threshold))
        
        # Response time score
        if metrics.response_time_p95 > self.policy.response_time_scale_up_threshold:
            scores.append(min(1.0, metrics.response_time_p95 / self.policy.response_time_scale_up_threshold))
        
        return max(scores) if scores else 0.0
    
    def _calculate_scale_down_score(self, metrics: ScalingMetrics) -> float:
        """Calculate scale down score (0-1)."""
        scores = []
        
        # CPU score (inverted - low usage = high score)
        if metrics.cpu_usage < self.policy.cpu_scale_down_threshold:
            scores.append(1.0 - (metrics.cpu_usage / self.policy.cpu_scale_down_threshold))
        
        # Memory score (inverted)
        if metrics.memory_usage < self.policy.memory_scale_down_threshold:
            scores.append(1.0 - (metrics.memory_usage / self.policy.memory_scale_down_threshold))
        
        # Queue score (inverted)
        if metrics.queue_length < self.policy.queue_scale_down_threshold:
            queue_ratio = metrics.queue_length / self.policy.queue_scale_down_threshold
            scores.append(1.0 - queue_ratio)
        
        return min(scores) if scores else 0.0
    
    @circuit_breaker("scaling_engine", CONFIGS["simulation"])
    def execute_scaling(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision."""
        try:
            old_instances = self.current_instances
            self.current_instances = decision.target_instances
            self.last_scaling_time = time.time()
            
            # Record scaling event
            scaling_event = {
                'timestamp': time.time(),
                'direction': decision.direction.value,
                'from_instances': old_instances,
                'to_instances': decision.target_instances,
                'reason': decision.reason,
                'confidence': decision.confidence
            }
            
            self.scaling_history.append(scaling_event)
            
            # Clear consecutive decisions after successful scaling
            self.consecutive_decisions.clear()
            
            logger.info(
                f"Scaling executed: {old_instances} -> {decision.target_instances} instances "
                f"({decision.direction.value}). Reason: {decision.reason}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
            return False
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        with self._lock:
            metrics = self.metrics_collector.get_current_metrics()
            
            time_since_last_scaling = time.time() - self.last_scaling_time if self.last_scaling_time > 0 else None
            
            return {
                'current_instances': self.current_instances,
                'policy': self.policy.name,
                'metrics': {
                    'cpu_usage': f"{metrics.cpu_usage:.1%}",
                    'memory_usage': f"{metrics.memory_usage:.1%}",
                    'agent_count': metrics.agent_count,
                    'steps_per_second': f"{metrics.steps_per_second:.1f}",
                    'queue_length': metrics.queue_length,
                    'response_time_p95': f"{metrics.response_time_p95:.2f}s",
                    'active_connections': metrics.active_connections,
                },
                'thresholds': {
                    'cpu_up': f"{self.policy.cpu_scale_up_threshold:.1%}",
                    'cpu_down': f"{self.policy.cpu_scale_down_threshold:.1%}",
                    'memory_up': f"{self.policy.memory_scale_up_threshold:.1%}",
                    'memory_down': f"{self.policy.memory_scale_down_threshold:.1%}",
                },
                'last_scaling_time': self.last_scaling_time,
                'time_since_last_scaling': time_since_last_scaling,
                'cooldown_remaining': max(0, self.policy.cooldown_period - (time_since_last_scaling or 0)),
                'consecutive_decisions': len(self.consecutive_decisions),
                'scaling_history_count': len(self.scaling_history),
            }


class AutoScaler:
    """High-level auto-scaler that manages multiple scaling engines."""
    
    def __init__(self):
        self.scaling_engines: Dict[str, ScalingEngine] = {}
        self.monitoring_thread = None
        self.monitoring_active = False
        self.scaling_callbacks: Dict[str, Callable] = {}
        
    def register_policy(self, name: str, policy: ScalingPolicy, 
                       scaling_callback: Optional[Callable] = None) -> None:
        """Register a scaling policy."""
        self.scaling_engines[name] = ScalingEngine(policy)
        if scaling_callback:
            self.scaling_callbacks[name] = scaling_callback
        
        logger.info(f"Registered scaling policy: {name}")
    
    def start_monitoring(self, evaluation_interval: float = 60.0) -> None:
        """Start auto-scaling monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(evaluation_interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Auto-scaler monitoring started (interval: {evaluation_interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop auto-scaling monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10.0)
        
        logger.info("Auto-scaler monitoring stopped")
    
    def _monitoring_loop(self, evaluation_interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                for name, engine in self.scaling_engines.items():
                    decision = engine.evaluate_scaling()
                    
                    if decision and decision.direction != ScalingDirection.NONE:
                        logger.info(f"Scaling decision for {name}: {decision.reason}")
                        
                        # Execute scaling
                        if engine.execute_scaling(decision):
                            # Call scaling callback if registered
                            if name in self.scaling_callbacks:
                                try:
                                    self.scaling_callbacks[name](decision)
                                except Exception as e:
                                    logger.error(f"Scaling callback failed for {name}: {e}")
                
            except Exception as e:
                logger.error(f"Auto-scaler monitoring error: {e}")
            
            time.sleep(evaluation_interval)
    
    def get_engine(self, name: str) -> Optional[ScalingEngine]:
        """Get scaling engine by name."""
        return self.scaling_engines.get(name)
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all scaling engines."""
        return {name: engine.get_scaling_status() 
                for name, engine in self.scaling_engines.items()}


# Predefined scaling policies
DEFAULT_POLICIES = {
    "conservative": ScalingPolicy(
        name="conservative",
        cpu_scale_up_threshold=0.80,
        cpu_scale_down_threshold=0.20,
        memory_scale_up_threshold=0.85,
        memory_scale_down_threshold=0.30,
        scale_up_factor=1.2,
        scale_down_factor=0.8,
        cooldown_period=600.0,  # 10 minutes
        consecutive_periods_required=5
    ),
    "aggressive": ScalingPolicy(
        name="aggressive",
        cpu_scale_up_threshold=0.60,
        cpu_scale_down_threshold=0.40,
        memory_scale_up_threshold=0.70,
        memory_scale_down_threshold=0.50,
        scale_up_factor=1.5,
        scale_down_factor=0.6,
        cooldown_period=180.0,  # 3 minutes
        consecutive_periods_required=2
    ),
    "simulation": ScalingPolicy(
        name="simulation",
        cpu_scale_up_threshold=0.75,
        memory_scale_up_threshold=0.80,
        queue_scale_up_threshold=50,
        response_time_scale_up_threshold=3.0,
        cpu_scale_down_threshold=0.30,
        memory_scale_down_threshold=0.40,
        queue_scale_down_threshold=5,
        scale_up_factor=1.3,
        scale_down_factor=0.7,
        cooldown_period=300.0,
        consecutive_periods_required=3
    )
}

# Global auto-scaler instance
auto_scaler = AutoScaler()