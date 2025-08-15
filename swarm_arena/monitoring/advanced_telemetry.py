"""
Advanced Telemetry Collection System
Real-time monitoring and metrics collection with alerting capabilities.
"""

import time
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import threading

from ..utils.logging import get_logger


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert configuration and state."""
    name: str
    metric: str
    threshold: float
    condition: str  # "greater_than", "less_than", "equals"
    enabled: bool = True
    triggered: bool = False
    trigger_count: int = 0
    last_triggered: Optional[float] = None
    callback: Optional[Callable] = None


class AdvancedTelemetryCollector:
    """Advanced telemetry system with real-time monitoring and alerting."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.retention_seconds = retention_hours * 3600
        self.logger = get_logger(__name__)
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Alerting
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # System state
        self.running = False
        self.collection_thread = None
        self.start_time = time.time()
        
        # Performance tracking
        self.performance_metrics = {
            'total_metrics_collected': 0,
            'alerts_triggered': 0,
            'collection_errors': 0
        }
    
    async def start(self) -> None:
        """Start telemetry collection."""
        if not self.running:
            self.running = True
            self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self.collection_thread.start()
            self.logger.info("Advanced telemetry collection started")
    
    async def stop(self) -> None:
        """Stop telemetry collection."""
        if self.running:
            self.running = False
            if self.collection_thread:
                self.collection_thread.join(timeout=5.0)
            self.logger.info("Advanced telemetry collection stopped")
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        try:
            if tags is None:
                tags = {}
            
            metric_point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags
            )
            
            self.metrics[name].append(metric_point)
            self.performance_metrics['total_metrics_collected'] += 1
            
            # Update aggregated metrics
            self._update_aggregated_metrics(name, value)
            
            # Check alerts
            self._check_alerts(name, value)
            
        except Exception as e:
            self.performance_metrics['collection_errors'] += 1
            self.logger.error(f"Failed to record metric {name}: {e}")
    
    async def record_arena_created(self, config: Any, response_time: float) -> None:
        """Record arena creation metrics."""
        self.record_metric("arena_creation_time", response_time, {"status": "success"})
        self.record_metric("arena_agent_count", config.num_agents)
        self.record_metric("arena_episodes", config.episode_length)
    
    async def record_simulation_completed(self, episodes: int, execution_time: float, results: Dict[str, Any]) -> None:
        """Record simulation completion metrics."""
        self.record_metric("simulation_execution_time", execution_time)
        self.record_metric("simulation_episodes", episodes)
        self.record_metric("simulation_error_rate", results.get('error_rate', 0.0))
        
        if 'mean_episode_time' in results:
            self.record_metric("episode_mean_time", results['mean_episode_time'])
    
    async def record_system_alert(self, alerts: List[str], context: Dict[str, Any]) -> None:
        """Record system alert event."""
        self.record_metric("system_alerts", len(alerts))
        
        for key, value in context.items():
            if isinstance(value, (int, float)):
                self.record_metric(f"alert_context_{key}", value)
    
    def add_alert(self, alert: Alert) -> None:
        """Add alert configuration."""
        self.alerts[alert.name] = alert
        self.logger.info(f"Alert added: {alert.name} on {alert.metric} {alert.condition} {alert.threshold}")
    
    def remove_alert(self, name: str) -> None:
        """Remove alert configuration."""
        if name in self.alerts:
            del self.alerts[name]
            self.logger.info(f"Alert removed: {name}")
    
    def get_metric_history(self, name: str, duration_seconds: Optional[int] = None) -> List[MetricPoint]:
        """Get metric history for specified duration."""
        if name not in self.metrics:
            return []
        
        metrics = list(self.metrics[name])
        
        if duration_seconds:
            cutoff_time = time.time() - duration_seconds
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return metrics
    
    def get_aggregated_metrics(self, name: str) -> Dict[str, float]:
        """Get aggregated statistics for a metric."""
        return self.aggregated_metrics.get(name, {})
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics and statistics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # System metrics
        system_metrics = {
            'uptime_seconds': uptime,
            'total_metrics': len(self.metrics),
            'total_data_points': sum(len(points) for points in self.metrics.values()),
            'collection_rate': self.performance_metrics['total_metrics_collected'] / max(uptime, 1),
            'alert_count': len([a for a in self.alerts.values() if a.enabled]),
            'active_alerts': len([a for a in self.alerts.values() if a.triggered])
        }
        
        # Recent metric summaries
        metric_summaries = {}
        for name, points in self.metrics.items():
            if points:
                recent_points = [p for p in points if current_time - p.timestamp < 300]  # Last 5 minutes
                if recent_points:
                    values = [p.value for p in recent_points]
                    metric_summaries[name] = {
                        'count': len(values),
                        'latest': values[-1],
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values)
                    }
        
        return {
            'system': system_metrics,
            'performance': self.performance_metrics,
            'metrics': metric_summaries,
            'alerts': {name: {'triggered': alert.triggered, 'trigger_count': alert.trigger_count} 
                      for name, alert in self.alerts.items()}
        }
    
    def _collection_loop(self) -> None:
        """Main collection loop for system metrics."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._cleanup_old_metrics()
                time.sleep(10.0)  # Collect every 10 seconds
                
            except Exception as e:
                self.performance_metrics['collection_errors'] += 1
                self.logger.error(f"Collection loop error: {e}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU and memory
            self.record_metric("system_cpu_percent", psutil.cpu_percent())
            
            memory = psutil.virtual_memory()
            self.record_metric("system_memory_percent", memory.percent)
            self.record_metric("system_memory_available_mb", memory.available / 1024 / 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.record_metric("system_disk_percent", disk.percent)
            
            # Process metrics
            process = psutil.Process()
            self.record_metric("process_memory_mb", process.memory_info().rss / 1024 / 1024)
            self.record_metric("process_cpu_percent", process.cpu_percent())
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
    
    def _update_aggregated_metrics(self, name: str, value: float) -> None:
        """Update aggregated statistics for a metric."""
        try:
            if name not in self.aggregated_metrics:
                self.aggregated_metrics[name] = {
                    'count': 0,
                    'sum': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'avg': 0.0
                }
            
            stats = self.aggregated_metrics[name]
            stats['count'] += 1
            stats['sum'] += value
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            stats['avg'] = stats['sum'] / stats['count']
            
        except Exception as e:
            self.logger.error(f"Failed to update aggregated metrics for {name}: {e}")
    
    def _check_alerts(self, metric_name: str, value: float) -> None:
        """Check if any alerts should be triggered."""
        try:
            for alert_name, alert in self.alerts.items():
                if not alert.enabled or alert.metric != metric_name:
                    continue
                
                should_trigger = False
                
                if alert.condition == "greater_than" and value > alert.threshold:
                    should_trigger = True
                elif alert.condition == "less_than" and value < alert.threshold:
                    should_trigger = True
                elif alert.condition == "equals" and abs(value - alert.threshold) < 0.001:
                    should_trigger = True
                
                if should_trigger and not alert.triggered:
                    # Trigger alert
                    alert.triggered = True
                    alert.trigger_count += 1
                    alert.last_triggered = time.time()
                    
                    self.performance_metrics['alerts_triggered'] += 1
                    
                    # Log alert
                    self.logger.warning(f"Alert triggered: {alert_name} - {metric_name} = {value} {alert.condition} {alert.threshold}")
                    
                    # Record in history
                    self.alert_history.append({
                        'timestamp': time.time(),
                        'alert_name': alert_name,
                        'metric': metric_name,
                        'value': value,
                        'threshold': alert.threshold,
                        'condition': alert.condition
                    })
                    
                    # Execute callback if provided
                    if alert.callback:
                        try:
                            alert.callback(alert, metric_name, value)
                        except Exception as e:
                            self.logger.error(f"Alert callback failed for {alert_name}: {e}")
                
                elif not should_trigger and alert.triggered:
                    # Reset alert
                    alert.triggered = False
                    self.logger.info(f"Alert cleared: {alert_name}")
                    
        except Exception as e:
            self.logger.error(f"Alert checking failed: {e}")
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metric data points."""
        try:
            cutoff_time = time.time() - self.retention_seconds
            
            for name, points in self.metrics.items():
                # Remove old points
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
                    
        except Exception as e:
            self.logger.error(f"Metric cleanup failed: {e}")
    
    def export_metrics(self, filepath: str, format: str = "json") -> None:
        """Export metrics to file."""
        try:
            export_data = {
                'export_timestamp': time.time(),
                'system_info': self.get_all_metrics(),
                'alerts': {
                    name: {
                        'metric': alert.metric,
                        'threshold': alert.threshold,
                        'condition': alert.condition,
                        'triggered': alert.triggered,
                        'trigger_count': alert.trigger_count
                    }
                    for name, alert in self.alerts.items()
                }
            }
            
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")


# Predefined alert configurations
def create_standard_alerts() -> List[Alert]:
    """Create standard system alerts."""
    return [
        Alert(
            name="high_cpu_usage",
            metric="system_cpu_percent",
            threshold=80.0,
            condition="greater_than"
        ),
        Alert(
            name="high_memory_usage", 
            metric="system_memory_percent",
            threshold=85.0,
            condition="greater_than"
        ),
        Alert(
            name="high_error_rate",
            metric="simulation_error_rate",
            threshold=10.0,
            condition="greater_than"
        ),
        Alert(
            name="slow_response_time",
            metric="arena_creation_time",
            threshold=5000.0,  # 5 seconds
            condition="greater_than"
        )
    ]