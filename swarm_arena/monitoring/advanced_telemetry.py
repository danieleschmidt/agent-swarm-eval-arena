"""Advanced telemetry and monitoring system."""

import time
import threading
import queue
import json
import statistics
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import logging
import psutil
import numpy as np


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class Metric:
    """A single metric measurement."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class Alert:
    """Alert definition and state."""
    name: str
    condition: Callable[[float], bool]
    message: str
    severity: str = "warning"
    cooldown: float = 300.0  # 5 minutes
    last_triggered: Optional[float] = None
    is_active: bool = False


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        self.lock = threading.Lock()
        
        # Alerts
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert, float], None]] = []
        
        # Auto-collection of system metrics
        self.system_metrics_enabled = True
        self.system_metrics_interval = 30.0  # seconds
        self._start_system_metrics_collection()
    
    def record_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric (cumulative).
        
        Args:
            name: Metric name
            value: Value to add
            tags: Additional tags
        """
        with self.lock:
            self.counters[name] += value
            
            metric = Metric(
                name=name,
                value=self.counters[name],
                metric_type=MetricType.COUNTER,
                tags=tags or {}
            )
            
            self.metrics[name].append(metric)
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric (instantaneous value).
        
        Args:
            name: Metric name
            value: Current value
            tags: Additional tags
        """
        with self.lock:
            self.gauges[name] = value
            
            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                tags=tags or {}
            )
            
            self.metrics[name].append(metric)
            
            # Check alerts
            self._check_alerts(name, value)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric.
        
        Args:
            name: Metric name
            value: Value to record
            tags: Additional tags
        """
        with self.lock:
            self.histograms[name].append(value)
            
            # Keep only recent values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-500:]
            
            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                tags=tags or {}
            )
            
            self.metrics[name].append(metric)
    
    def time_function(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Decorator to time function execution.
        
        Args:
            name: Metric name
            tags: Additional tags
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.record_timer(name, duration, tags)
            return wrapper
        return decorator
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric.
        
        Args:
            name: Metric name
            duration: Duration in seconds
            tags: Additional tags
        """
        with self.lock:
            self.timers[name].append(duration)
            
            # Keep only recent values
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-500:]
            
            metric = Metric(
                name=name,
                value=duration,
                metric_type=MetricType.TIMER,
                tags=tags or {},
                unit="seconds"
            )
            
            self.metrics[name].append(metric)
    
    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        return self.counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value."""
        return self.gauges.get(name)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics.
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary with histogram statistics
        """
        if name not in self.histograms or not self.histograms[name]:
            return {}
        
        values = self.histograms[name]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'std': statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics."""
        if name not in self.timers or not self.timers[name]:
            return {}
        
        values = self.timers[name]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'total': sum(values)
        }
    
    def get_rate(self, name: str, window_seconds: float = 60.0) -> float:
        """Calculate rate of events per second.
        
        Args:
            name: Metric name
            window_seconds: Time window for rate calculation
            
        Returns:
            Events per second
        """
        if name not in self.metrics:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Count events in time window
        count = 0
        for metric in reversed(self.metrics[name]):
            if metric.timestamp >= cutoff_time:
                count += 1
            else:
                break
        
        return count / window_seconds
    
    def add_alert(self, alert: Alert) -> None:
        """Add an alert condition.
        
        Args:
            alert: Alert configuration
        """
        self.alerts[alert.name] = alert
    
    def add_alert_handler(self, handler: Callable[[Alert, float], None]) -> None:
        """Add an alert handler function.
        
        Args:
            handler: Function to call when alert triggers
        """
        self.alert_handlers.append(handler)
    
    def _check_alerts(self, metric_name: str, value: float) -> None:
        """Check if any alerts should trigger."""
        current_time = time.time()
        
        for alert in self.alerts.values():
            # Skip if in cooldown
            if (alert.last_triggered and 
                current_time - alert.last_triggered < alert.cooldown):
                continue
            
            # Check if condition is met
            if alert.condition(value):
                alert.last_triggered = current_time
                alert.is_active = True
                
                # Trigger alert handlers
                for handler in self.alert_handlers:
                    try:
                        handler(alert, value)
                    except Exception as e:
                        logging.error(f"Alert handler failed: {e}")
            else:
                alert.is_active = False
    
    def _start_system_metrics_collection(self) -> None:
        """Start automatic system metrics collection."""
        def collect_system_metrics():
            while self.system_metrics_enabled:
                try:
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.record_gauge("system.cpu.percent", cpu_percent)
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.record_gauge("system.memory.percent", memory.percent)
                    self.record_gauge("system.memory.available", memory.available)
                    self.record_gauge("system.memory.used", memory.used)
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    self.record_gauge("system.disk.percent", disk.percent)
                    self.record_gauge("system.disk.free", disk.free)
                    self.record_gauge("system.disk.used", disk.used)
                    
                    # Network metrics
                    network = psutil.net_io_counters()
                    self.record_counter("system.network.bytes_sent", network.bytes_sent)
                    self.record_counter("system.network.bytes_recv", network.bytes_recv)
                    
                    time.sleep(self.system_metrics_interval)
                    
                except Exception as e:
                    logging.error(f"System metrics collection failed: {e}")
                    time.sleep(self.system_metrics_interval)
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        with self.lock:
            summary = {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {},
                'timers': {},
                'rates': {},
                'alerts': {
                    name: {
                        'is_active': alert.is_active,
                        'last_triggered': alert.last_triggered,
                        'severity': alert.severity
                    }
                    for name, alert in self.alerts.items()
                }
            }
            
            # Add histogram stats
            for name in self.histograms:
                summary['histograms'][name] = self.get_histogram_stats(name)
            
            # Add timer stats
            for name in self.timers:
                summary['timers'][name] = self.get_timer_stats(name)
            
            # Add rates
            for name in self.metrics:
                summary['rates'][name] = self.get_rate(name)
        
        return summary
    
    def export_to_prometheus(self) -> str:
        """Export metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        lines = []
        
        # Counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # Histograms
        for name, values in self.histograms.items():
            if values:
                stats = self.get_histogram_stats(name)
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_sum {sum(values)}")
                
                # Add quantiles
                for quantile in [0.5, 0.95, 0.99]:
                    q_value = np.percentile(values, quantile * 100)
                    lines.append(f"{name}{{quantile=\"{quantile}\"}} {q_value}")
        
        return '\n'.join(lines)


class PerformanceProfiler:
    """Performance profiling and analysis."""
    
    def __init__(self):
        self.profiles: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.active_profiles: Dict[str, float] = {}
        
    def start_profile(self, name: str) -> None:
        """Start profiling a section.
        
        Args:
            name: Profile section name
        """
        self.active_profiles[name] = time.time()
    
    def end_profile(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """End profiling a section.
        
        Args:
            name: Profile section name
            metadata: Additional metadata
            
        Returns:
            Duration in seconds
        """
        if name not in self.active_profiles:
            raise ValueError(f"Profile '{name}' not started")
        
        duration = time.time() - self.active_profiles[name]
        del self.active_profiles[name]
        
        profile_data = {
            'duration': duration,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.profiles[name].append(profile_data)
        
        # Keep only recent profiles
        if len(self.profiles[name]) > 1000:
            self.profiles[name] = self.profiles[name][-500:]
        
        return duration
    
    def profile_context(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for profiling.
        
        Args:
            name: Profile section name
            metadata: Additional metadata
        """
        class ProfileContext:
            def __init__(self, profiler, name, metadata):
                self.profiler = profiler
                self.name = name
                self.metadata = metadata
                self.duration = 0
            
            def __enter__(self):
                self.profiler.start_profile(self.name)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.duration = self.profiler.end_profile(self.name, self.metadata)
        
        return ProfileContext(self, name, metadata)
    
    def get_profile_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a profile.
        
        Args:
            name: Profile name
            
        Returns:
            Profile statistics
        """
        if name not in self.profiles or not self.profiles[name]:
            return {}
        
        durations = [p['duration'] for p in self.profiles[name]]
        
        return {
            'count': len(durations),
            'total_time': sum(durations),
            'min_time': min(durations),
            'max_time': max(durations),
            'mean_time': statistics.mean(durations),
            'median_time': statistics.median(durations),
            'p95_time': np.percentile(durations, 95),
            'p99_time': np.percentile(durations, 99),
            'std_time': statistics.stdev(durations) if len(durations) > 1 else 0
        }
    
    def get_slowest_profiles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest profile sections.
        
        Args:
            limit: Number of results to return
            
        Returns:
            List of slowest profiles
        """
        all_profiles = []
        
        for name, profiles in self.profiles.items():
            for profile in profiles:
                all_profiles.append({
                    'name': name,
                    'duration': profile['duration'],
                    'timestamp': profile['timestamp'],
                    'metadata': profile['metadata']
                })
        
        # Sort by duration and return top results
        all_profiles.sort(key=lambda x: x['duration'], reverse=True)
        return all_profiles[:limit]


class EventTracker:
    """Tracks events and patterns in the system."""
    
    def __init__(self, max_events: int = 100000):
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
    
    def track_event(self, 
                   event_type: str, 
                   data: Optional[Dict[str, Any]] = None,
                   tags: Optional[Dict[str, str]] = None) -> None:
        """Track an event.
        
        Args:
            event_type: Type of event
            data: Event data
            tags: Event tags
        """
        with self.lock:
            event = {
                'type': event_type,
                'timestamp': time.time(),
                'data': data or {},
                'tags': tags or {}
            }
            
            self.events.append(event)
            self.event_counts[event_type] += 1
    
    def get_events(self, 
                   event_type: Optional[str] = None,
                   since: Optional[float] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get events matching criteria.
        
        Args:
            event_type: Filter by event type
            since: Filter events since timestamp
            limit: Maximum number of events
            
        Returns:
            List of matching events
        """
        with self.lock:
            filtered_events = []
            
            for event in reversed(self.events):
                # Apply filters
                if event_type and event['type'] != event_type:
                    continue
                
                if since and event['timestamp'] < since:
                    break
                
                filtered_events.append(event)
                
                if limit and len(filtered_events) >= limit:
                    break
            
            return list(reversed(filtered_events))
    
    def get_event_counts(self, 
                        since: Optional[float] = None) -> Dict[str, int]:
        """Get event counts.
        
        Args:
            since: Count events since timestamp
            
        Returns:
            Dictionary mapping event types to counts
        """
        if since is None:
            return dict(self.event_counts)
        
        counts = defaultdict(int)
        
        with self.lock:
            for event in self.events:
                if event['timestamp'] >= since:
                    counts[event['type']] += 1
        
        return dict(counts)
    
    def detect_anomalies(self, 
                        window_minutes: float = 10.0,
                        threshold_factor: float = 3.0) -> List[Dict[str, Any]]:
        """Detect anomalous event patterns.
        
        Args:
            window_minutes: Time window for analysis
            threshold_factor: Factor above normal for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        window_seconds = window_minutes * 60
        current_time = time.time()
        
        # Get baseline rates
        baseline_start = current_time - (window_seconds * 4)  # 4x window for baseline
        baseline_end = current_time - window_seconds
        
        baseline_counts = self.get_event_counts(since=baseline_start)
        baseline_duration = baseline_end - baseline_start
        
        # Get current window rates
        current_counts = self.get_event_counts(since=current_time - window_seconds)
        
        anomalies = []
        
        for event_type in set(list(baseline_counts.keys()) + list(current_counts.keys())):
            baseline_rate = baseline_counts.get(event_type, 0) / baseline_duration
            current_rate = current_counts.get(event_type, 0) / window_seconds
            
            # Check for anomalies
            if current_rate > baseline_rate * threshold_factor:
                anomalies.append({
                    'event_type': event_type,
                    'baseline_rate': baseline_rate,
                    'current_rate': current_rate,
                    'factor': current_rate / baseline_rate if baseline_rate > 0 else float('inf'),
                    'window_minutes': window_minutes
                })
        
        return anomalies


# Global instances
metrics_collector = MetricsCollector()
performance_profiler = PerformanceProfiler()
event_tracker = EventTracker()


# Convenience functions

def record_metric(name: str, value: float, metric_type: MetricType = MetricType.GAUGE, 
                 tags: Optional[Dict[str, str]] = None) -> None:
    """Record a metric value."""
    if metric_type == MetricType.COUNTER:
        metrics_collector.record_counter(name, value, tags)
    elif metric_type == MetricType.GAUGE:
        metrics_collector.record_gauge(name, value, tags)
    elif metric_type == MetricType.HISTOGRAM:
        metrics_collector.record_histogram(name, value, tags)
    elif metric_type == MetricType.TIMER:
        metrics_collector.record_timer(name, value, tags)


def time_operation(name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to time an operation."""
    return metrics_collector.time_function(name, tags)


def profile_operation(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager to profile an operation."""
    return performance_profiler.profile_context(name, metadata)


def track_event(event_type: str, data: Optional[Dict[str, Any]] = None,
               tags: Optional[Dict[str, str]] = None) -> None:
    """Track an event."""
    event_tracker.track_event(event_type, data, tags)