"""Health monitoring and recovery system for swarm arena."""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels for components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    threshold_warn: float
    threshold_critical: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def status(self) -> HealthStatus:
        """Get health status based on thresholds."""
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warn:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


@dataclass
class ComponentHealth:
    """Health status for a system component."""
    component_name: str
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)
    recovery_actions: Dict[str, Callable] = field(default_factory=dict)
    
    @property
    def overall_status(self) -> HealthStatus:
        """Get overall component health status."""
        if not self.metrics:
            return HealthStatus.HEALTHY
        
        statuses = [metric.status for metric in self.metrics.values()]
        
        if HealthStatus.FAILED in statuses:
            return HealthStatus.FAILED
        elif HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def add_metric(self, metric: HealthMetric) -> None:
        """Add or update a health metric."""
        self.metrics[metric.name] = metric
        self.last_update = time.time()
    
    def get_unhealthy_metrics(self) -> List[HealthMetric]:
        """Get list of metrics that are not healthy."""
        return [
            metric for metric in self.metrics.values()
            if metric.status != HealthStatus.HEALTHY
        ]


class HealthMonitor:
    """Advanced health monitoring system with automatic recovery."""
    
    def __init__(self, check_interval: float = 1.0):
        """Initialize health monitor.
        
        Args:
            check_interval: Time between health checks in seconds
        """
        self.check_interval = check_interval
        self.components: Dict[str, ComponentHealth] = {}
        self.global_health_history = deque(maxlen=1000)
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.recovery_callbacks: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        
        # Register default system metrics
        self._register_default_components()
    
    def _register_default_components(self) -> None:
        """Register default system components for monitoring."""
        # Arena performance metrics
        self.register_component("arena_performance")
        self.register_component("memory_usage")
        self.register_component("agent_health")
        self.register_component("environment_stability")
        
    def register_component(self, component_name: str, 
                         recovery_actions: Optional[Dict[str, Callable]] = None) -> None:
        """Register a component for health monitoring.
        
        Args:
            component_name: Name of the component
            recovery_actions: Dictionary of recovery action names to callables
        """
        with self._lock:
            self.components[component_name] = ComponentHealth(
                component_name=component_name,
                recovery_actions=recovery_actions or {}
            )
            self.recovery_callbacks[component_name] = []
    
    def add_recovery_callback(self, component_name: str, callback: Callable) -> None:
        """Add recovery callback for a component.
        
        Args:
            component_name: Name of the component
            callback: Callback function to execute on recovery
        """
        with self._lock:
            if component_name not in self.recovery_callbacks:
                self.recovery_callbacks[component_name] = []
            self.recovery_callbacks[component_name].append(callback)
    
    def update_metric(self, component_name: str, metric_name: str, 
                     value: float, threshold_warn: float, 
                     threshold_critical: float) -> None:
        """Update a health metric for a component.
        
        Args:
            component_name: Name of the component
            metric_name: Name of the metric
            value: Current metric value
            threshold_warn: Warning threshold
            threshold_critical: Critical threshold
        """
        with self._lock:
            if component_name not in self.components:
                self.register_component(component_name)
            
            metric = HealthMetric(
                name=metric_name,
                value=value,
                threshold_warn=threshold_warn,
                threshold_critical=threshold_critical
            )
            
            self.components[component_name].add_metric(metric)
            
            # Trigger recovery if needed
            self._check_and_trigger_recovery(component_name, metric)
    
    def _check_and_trigger_recovery(self, component_name: str, 
                                   metric: HealthMetric) -> None:
        """Check if recovery is needed and trigger it."""
        if metric.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
            logger.warning(f"Health issue detected: {component_name}.{metric.name} = {metric.value}")
            self._trigger_recovery(component_name, metric)
    
    def _trigger_recovery(self, component_name: str, metric: HealthMetric) -> None:
        """Trigger recovery actions for a component."""
        try:
            component = self.components[component_name]
            
            # Execute registered recovery actions
            if metric.name in component.recovery_actions:
                logger.info(f"Executing recovery action for {component_name}.{metric.name}")
                component.recovery_actions[metric.name]()
            
            # Execute recovery callbacks
            for callback in self.recovery_callbacks.get(component_name, []):
                try:
                    callback(component_name, metric)
                except Exception as e:
                    logger.error(f"Recovery callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Recovery action failed for {component_name}: {e}")
    
    def start_monitoring(self) -> None:
        """Start the health monitoring thread."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the health monitoring thread."""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                self._perform_health_check()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _perform_health_check(self) -> None:
        """Perform comprehensive health check."""
        try:
            # Update system metrics
            self._update_system_metrics()
            
            # Calculate global health score
            global_score = self._calculate_global_health_score()
            self.global_health_history.append({
                'timestamp': time.time(),
                'score': global_score,
                'component_count': len(self.components)
            })
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _update_system_metrics(self) -> None:
        """Update built-in system health metrics."""
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.update_metric(
                "memory_usage", "memory_percent", 
                memory.percent, 80.0, 95.0
            )
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.update_metric(
                "arena_performance", "cpu_percent",
                cpu_percent, 80.0, 95.0
            )
            
        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            logger.warning(f"System metrics update failed: {e}")
    
    def _calculate_global_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        if not self.components:
            return 100.0
        
        component_scores = []
        for component in self.components.values():
            if not component.metrics:
                component_scores.append(100.0)
                continue
            
            metric_scores = []
            for metric in component.metrics.values():
                if metric.status == HealthStatus.HEALTHY:
                    metric_scores.append(100.0)
                elif metric.status == HealthStatus.DEGRADED:
                    metric_scores.append(70.0)
                elif metric.status == HealthStatus.CRITICAL:
                    metric_scores.append(30.0)
                else:  # FAILED
                    metric_scores.append(0.0)
            
            component_scores.append(np.mean(metric_scores))
        
        return np.mean(component_scores)
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        with self._lock:
            report = {
                'timestamp': time.time(),
                'global_health_score': self._calculate_global_health_score(),
                'components': {},
                'unhealthy_components': [],
                'recommendations': []
            }
            
            for name, component in self.components.items():
                component_data = {
                    'status': component.overall_status.value,
                    'metrics': {},
                    'last_update': component.last_update
                }
                
                for metric_name, metric in component.metrics.items():
                    component_data['metrics'][metric_name] = {
                        'value': metric.value,
                        'status': metric.status.value,
                        'threshold_warn': metric.threshold_warn,
                        'threshold_critical': metric.threshold_critical
                    }
                
                report['components'][name] = component_data
                
                # Track unhealthy components
                if component.overall_status != HealthStatus.HEALTHY:
                    report['unhealthy_components'].append({
                        'name': name,
                        'status': component.overall_status.value,
                        'unhealthy_metrics': [
                            metric.name for metric in component.get_unhealthy_metrics()
                        ]
                    })
            
            # Generate recommendations
            report['recommendations'] = self._generate_health_recommendations(report)
            
            return report
    
    def _generate_health_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if report['global_health_score'] < 50:
            recommendations.append("System health is critical - immediate action required")
        elif report['global_health_score'] < 80:
            recommendations.append("System health is degraded - review unhealthy components")
        
        for component_info in report['unhealthy_components']:
            component_name = component_info['name']
            unhealthy_metrics = component_info['unhealthy_metrics']
            
            if 'memory_percent' in unhealthy_metrics:
                recommendations.append(f"High memory usage in {component_name} - consider optimization")
            
            if 'cpu_percent' in unhealthy_metrics:
                recommendations.append(f"High CPU usage in {component_name} - review performance")
            
            if 'error_rate' in unhealthy_metrics:
                recommendations.append(f"High error rate in {component_name} - investigate failures")
        
        return recommendations
    
    def get_health_trends(self, duration_seconds: int = 300) -> Dict[str, Any]:
        """Get health trends over specified duration."""
        cutoff_time = time.time() - duration_seconds
        recent_history = [
            entry for entry in self.global_health_history
            if entry['timestamp'] >= cutoff_time
        ]
        
        if len(recent_history) < 2:
            return {'trend': 'insufficient_data', 'data_points': len(recent_history)}
        
        scores = [entry['score'] for entry in recent_history]
        timestamps = [entry['timestamp'] for entry in recent_history]
        
        # Calculate trend
        if len(scores) >= 2:
            trend_slope = (scores[-1] - scores[0]) / (timestamps[-1] - timestamps[0])
            if trend_slope > 1:
                trend = 'improving'
            elif trend_slope < -1:
                trend = 'degrading'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'
        
        return {
            'trend': trend,
            'current_score': scores[-1] if scores else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'avg_score': np.mean(scores) if scores else 0,
            'data_points': len(recent_history),
            'time_range': duration_seconds
        }
    
    def is_system_healthy(self, min_score: float = 70.0) -> bool:
        """Check if system is healthy above minimum threshold."""
        current_score = self._calculate_global_health_score()
        return current_score >= min_score
    
    def wait_for_healthy_state(self, timeout: float = 30.0, 
                              min_score: float = 70.0) -> bool:
        """Wait for system to reach healthy state.
        
        Args:
            timeout: Maximum time to wait in seconds
            min_score: Minimum health score to consider healthy
            
        Returns:
            True if system became healthy within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_system_healthy(min_score):
                return True
            time.sleep(0.5)
        
        return False