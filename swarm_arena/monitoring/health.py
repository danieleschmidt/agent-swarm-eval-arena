"""Health monitoring and system diagnostics."""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque

from ..utils.logging import get_logger
from ..utils.error_handler import error_manager

logger = get_logger(__name__)


@dataclass
class HealthMetrics:
    """System health metrics snapshot."""
    
    timestamp: float = field(default_factory=time.time)
    
    # System metrics
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_percent: float = 0.0
    
    # Application metrics
    active_agents: int = 0
    total_agents: int = 0
    simulation_fps: float = 0.0
    spatial_index_queries: int = 0
    telemetry_data_points: int = 0
    
    # Error metrics
    total_errors: int = 0
    recent_errors: int = 0
    error_recovery_rate: float = 1.0
    critical_errors: int = 0
    
    # Performance metrics  
    avg_step_time_ms: float = 0.0
    memory_usage_trend: str = "stable"  # stable, increasing, decreasing
    cpu_usage_trend: str = "stable"
    
    # Health status
    overall_status: str = "healthy"  # healthy, warning, critical
    status_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "system": {
                "cpu_percent": self.cpu_percent,
                "memory_percent": self.memory_percent,
                "memory_available_mb": self.memory_available_mb,
                "disk_usage_percent": self.disk_usage_percent
            },
            "application": {
                "active_agents": self.active_agents,
                "total_agents": self.total_agents,
                "simulation_fps": self.simulation_fps,
                "spatial_index_queries": self.spatial_index_queries,
                "telemetry_data_points": self.telemetry_data_points
            },
            "errors": {
                "total_errors": self.total_errors,
                "recent_errors": self.recent_errors,
                "error_recovery_rate": self.error_recovery_rate,
                "critical_errors": self.critical_errors
            },
            "performance": {
                "avg_step_time_ms": self.avg_step_time_ms,
                "memory_usage_trend": self.memory_usage_trend,
                "cpu_usage_trend": self.cpu_usage_trend
            },
            "status": {
                "overall_status": self.overall_status,
                "status_reasons": self.status_reasons
            }
        }


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, 
                 check_interval: float = 5.0,
                 history_size: int = 100,
                 auto_start: bool = True):
        """Initialize health monitor.
        
        Args:
            check_interval: Seconds between health checks
            history_size: Number of health snapshots to keep
            auto_start: Whether to start monitoring automatically
        """
        self.check_interval = check_interval
        self.history_size = history_size
        
        # Health data
        self.health_history: deque = deque(maxlen=history_size)
        self.current_metrics = HealthMetrics()
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        # Health check callbacks
        self.health_callbacks: List[Callable[[HealthMetrics], None]] = []
        
        # Alert thresholds
        self.thresholds = {
            "cpu_warning": 70.0,
            "cpu_critical": 90.0,
            "memory_warning": 80.0,
            "memory_critical": 95.0,
            "disk_warning": 85.0,
            "disk_critical": 95.0,
            "error_rate_warning": 5.0,  # errors per minute
            "error_rate_critical": 15.0,
            "step_time_warning": 100.0,  # milliseconds
            "step_time_critical": 500.0
        }
        
        # Trend analysis
        self.trend_window = 10  # Number of samples for trend analysis
        
        if auto_start:
            self.start_monitoring()
        
        logger.info("Health monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self.monitoring:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        logger.info("Health monitoring stopped")
    
    def add_health_callback(self, callback: Callable[[HealthMetrics], None]) -> None:
        """Add callback for health updates.
        
        Args:
            callback: Function to call with health metrics
        """
        self.health_callbacks.append(callback)
        logger.debug(f"Added health callback: {callback.__name__}")
    
    def remove_health_callback(self, callback: Callable[[HealthMetrics], None]) -> None:
        """Remove health callback.
        
        Args:
            callback: Callback to remove
        """
        if callback in self.health_callbacks:
            self.health_callbacks.remove(callback)
            logger.debug(f"Removed health callback: {callback.__name__}")
    
    def collect_health_metrics(self, arena=None, telemetry=None) -> HealthMetrics:
        """Collect current health metrics.
        
        Args:
            arena: Arena instance for application metrics
            telemetry: Telemetry collector for telemetry metrics
            
        Returns:
            Current health metrics
        """
        try:
            metrics = HealthMetrics()
            
            # System metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            metrics.memory_percent = memory_info.percent
            metrics.memory_available_mb = memory_info.available / 1024 / 1024
            
            try:
                disk_info = psutil.disk_usage('/')
                metrics.disk_usage_percent = disk_info.percent
            except Exception:
                metrics.disk_usage_percent = 0.0
            
            # Application metrics from arena
            if arena:
                metrics.active_agents = sum(1 for agent in arena.agents.values() if agent.state.alive)
                metrics.total_agents = len(arena.agents)
                
                # Calculate FPS from step times
                if arena.step_times:
                    recent_times = arena.step_times[-10:]  # Last 10 steps
                    avg_step_time = sum(recent_times) / len(recent_times)
                    metrics.simulation_fps = 1.0 / max(avg_step_time, 0.001)
                    metrics.avg_step_time_ms = avg_step_time * 1000
                
                # Spatial index metrics
                if hasattr(arena, 'spatial_index'):
                    spatial_stats = arena.spatial_index.get_statistics()
                    metrics.spatial_index_queries = spatial_stats.get("query_count", 0)
            
            # Telemetry metrics
            if telemetry:
                telemetry_stats = telemetry.get_statistics()
                if isinstance(telemetry_stats, dict) and "data_points" in telemetry_stats:
                    metrics.telemetry_data_points = telemetry_stats["data_points"]
            
            # Error metrics
            error_stats = error_manager.get_error_statistics()
            if isinstance(error_stats, dict):
                metrics.total_errors = error_stats.get("total_errors", 0)
                metrics.recent_errors = error_stats.get("recent_errors", 0)
                metrics.error_recovery_rate = error_stats.get("recovery_rate", 1.0)
                metrics.critical_errors = error_stats.get("critical_errors", 0)
            
            # Analyze trends
            self._analyze_trends(metrics)
            
            # Determine overall health status
            self._determine_health_status(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect health metrics: {str(e)}")
            # Return minimal metrics on error
            error_metrics = HealthMetrics()
            error_metrics.overall_status = "warning"
            error_metrics.status_reasons = ["Health collection failed"]
            return error_metrics
    
    def _analyze_trends(self, current_metrics: HealthMetrics) -> None:
        """Analyze metric trends over recent history.
        
        Args:
            current_metrics: Current metrics to update with trends
        """
        if len(self.health_history) < self.trend_window:
            return  # Not enough data for trend analysis
        
        recent_history = list(self.health_history)[-self.trend_window:]
        
        # CPU trend
        cpu_values = [m.cpu_percent for m in recent_history]
        cpu_trend = self._calculate_trend(cpu_values)
        current_metrics.cpu_usage_trend = cpu_trend
        
        # Memory trend  
        memory_values = [m.memory_percent for m in recent_history]
        memory_trend = self._calculate_trend(memory_values)
        current_metrics.memory_usage_trend = memory_trend
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from list of values.
        
        Args:
            values: List of metric values
            
        Returns:
            Trend string: "increasing", "decreasing", or "stable"
        """
        if len(values) < 3:
            return "stable"
        
        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        # Classify trend
        if slope > 1.0:  # Significant positive slope
            return "increasing"
        elif slope < -1.0:  # Significant negative slope
            return "decreasing"
        else:
            return "stable"
    
    def _determine_health_status(self, metrics: HealthMetrics) -> None:
        """Determine overall health status and reasons.
        
        Args:
            metrics: Metrics to analyze
        """
        status_reasons = []
        critical_issues = 0
        warning_issues = 0
        
        # Check system resources
        if metrics.cpu_percent >= self.thresholds["cpu_critical"]:
            status_reasons.append(f"Critical CPU usage: {metrics.cpu_percent:.1f}%")
            critical_issues += 1
        elif metrics.cpu_percent >= self.thresholds["cpu_warning"]:
            status_reasons.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            warning_issues += 1
        
        if metrics.memory_percent >= self.thresholds["memory_critical"]:
            status_reasons.append(f"Critical memory usage: {metrics.memory_percent:.1f}%")
            critical_issues += 1
        elif metrics.memory_percent >= self.thresholds["memory_warning"]:
            status_reasons.append(f"High memory usage: {metrics.memory_percent:.1f}%")
            warning_issues += 1
        
        if metrics.disk_usage_percent >= self.thresholds["disk_critical"]:
            status_reasons.append(f"Critical disk usage: {metrics.disk_usage_percent:.1f}%")
            critical_issues += 1
        elif metrics.disk_usage_percent >= self.thresholds["disk_warning"]:
            status_reasons.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
            warning_issues += 1
        
        # Check performance
        if metrics.avg_step_time_ms >= self.thresholds["step_time_critical"]:
            status_reasons.append(f"Critical step time: {metrics.avg_step_time_ms:.1f}ms")
            critical_issues += 1
        elif metrics.avg_step_time_ms >= self.thresholds["step_time_warning"]:
            status_reasons.append(f"Slow performance: {metrics.avg_step_time_ms:.1f}ms")
            warning_issues += 1
        
        # Check errors
        if metrics.critical_errors > 0:
            status_reasons.append(f"Critical errors present: {metrics.critical_errors}")
            critical_issues += 1
        
        error_rate = metrics.recent_errors / 5.0  # Per minute (5 minute window)
        if error_rate >= self.thresholds["error_rate_critical"]:
            status_reasons.append(f"Critical error rate: {error_rate:.1f}/min")
            critical_issues += 1
        elif error_rate >= self.thresholds["error_rate_warning"]:
            status_reasons.append(f"High error rate: {error_rate:.1f}/min")
            warning_issues += 1
        
        # Check trends
        if metrics.cpu_usage_trend == "increasing" and metrics.cpu_percent > 50:
            status_reasons.append("CPU usage trending upward")
            warning_issues += 1
        
        if metrics.memory_usage_trend == "increasing" and metrics.memory_percent > 60:
            status_reasons.append("Memory usage trending upward")
            warning_issues += 1
        
        # Determine overall status
        if critical_issues > 0:
            metrics.overall_status = "critical"
        elif warning_issues > 0:
            metrics.overall_status = "warning"
        else:
            metrics.overall_status = "healthy"
        
        metrics.status_reasons = status_reasons
    
    def get_current_health(self) -> HealthMetrics:
        """Get current health metrics.
        
        Returns:
            Current health metrics
        """
        with self.lock:
            return self.current_metrics
    
    def get_health_history(self, last_n: Optional[int] = None) -> List[HealthMetrics]:
        """Get health history.
        
        Args:
            last_n: Number of recent entries (None for all)
            
        Returns:
            List of health metrics
        """
        with self.lock:
            history = list(self.health_history)
            if last_n is not None:
                history = history[-last_n:]
            return history
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary with key indicators.
        
        Returns:
            Health summary dictionary
        """
        with self.lock:
            current = self.current_metrics
            
            return {
                "status": current.overall_status,
                "timestamp": current.timestamp,
                "key_metrics": {
                    "cpu_percent": current.cpu_percent,
                    "memory_percent": current.memory_percent,
                    "simulation_fps": current.simulation_fps,
                    "active_agents": current.active_agents,
                    "recent_errors": current.recent_errors
                },
                "issues": current.status_reasons,
                "trends": {
                    "cpu": current.cpu_usage_trend,
                    "memory": current.memory_usage_trend
                }
            }
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        logger.debug("Health monitoring loop started")
        
        while self.monitoring:
            try:
                # Collect metrics (will be updated by external systems)
                # This is a basic collection, real metrics come from update_health_metrics
                basic_metrics = HealthMetrics()
                basic_metrics.cpu_percent = psutil.cpu_percent(interval=None)
                basic_metrics.memory_percent = psutil.virtual_memory().percent
                
                with self.lock:
                    self.current_metrics = basic_metrics
                    self.health_history.append(basic_metrics)
                
                # Call health callbacks
                for callback in self.health_callbacks:
                    try:
                        callback(basic_metrics)
                    except Exception as e:
                        logger.warning(f"Health callback {callback.__name__} failed: {str(e)}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring loop error: {str(e)}")
                time.sleep(self.check_interval)
        
        logger.debug("Health monitoring loop stopped")
    
    def update_health_metrics(self, arena=None, telemetry=None) -> None:
        """Update health metrics with current system state.
        
        Args:
            arena: Arena instance for application metrics
            telemetry: Telemetry collector instance
        """
        metrics = self.collect_health_metrics(arena, telemetry)
        
        with self.lock:
            self.current_metrics = metrics
            self.health_history.append(metrics)
    
    def cleanup(self) -> None:
        """Cleanup health monitor resources."""
        try:
            self.stop_monitoring()
            self.health_history.clear()
            self.health_callbacks.clear()
            logger.info("Health monitor cleaned up")
        except Exception as e:
            logger.warning(f"Health monitor cleanup error: {str(e)}")
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass