"""Monitoring and health check utilities for the Swarm Arena."""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from ..exceptions import SimulationError
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    fps: float = 0.0
    step_time: float = 0.0
    agents_active: int = 0
    resources_available: int = 0
    errors_count: int = 0


@dataclass
class HealthStatus:
    """System health status."""
    
    timestamp: float = field(default_factory=time.time)
    status: str = "healthy"  # "healthy", "warning", "critical"
    cpu_ok: bool = True
    memory_ok: bool = True
    performance_ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PerformanceMonitor:
    """Monitor system performance and simulation health."""
    
    def __init__(self, 
                 cpu_threshold: float = 90.0,
                 memory_threshold: float = 85.0,
                 min_fps: float = 10.0,
                 max_step_time: float = 1.0) -> None:
        """Initialize performance monitor.
        
        Args:
            cpu_threshold: CPU usage threshold (%)
            memory_threshold: Memory usage threshold (%)
            min_fps: Minimum acceptable FPS
            max_step_time: Maximum acceptable step time (seconds)
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.min_fps = min_fps
        self.max_step_time = max_step_time
        
        self.metrics_history: List[PerformanceMetrics] = []
        self.health_history: List[HealthStatus] = []
        self.error_count = 0
        self.warning_count = 0
        
        # Threading for continuous monitoring
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start continuous performance monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            logger.warning("Performance monitoring already active")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started performance monitoring with {interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop continuous performance monitoring."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped performance monitoring")
    
    def record_metrics(self, 
                      fps: float = 0.0,
                      step_time: float = 0.0,
                      agents_active: int = 0,
                      resources_available: int = 0) -> PerformanceMetrics:
        """Record current performance metrics.
        
        Args:
            fps: Current frames per second
            step_time: Current step execution time
            agents_active: Number of active agents
            resources_available: Number of available resources
            
        Returns:
            Recorded metrics
        """
        try:
            with self.lock:
                # Get system metrics
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                memory_usage = memory_info.percent
                
                # Create metrics
                metrics = PerformanceMetrics(
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    fps=fps,
                    step_time=step_time,
                    agents_active=agents_active,
                    resources_available=resources_available,
                    errors_count=self.error_count
                )
                
                # Store metrics (keep last 1000 entries)
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 1000:
                    self.metrics_history.pop(0)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to record metrics: {str(e)}")
            self.error_count += 1
            return PerformanceMetrics()
    
    def check_health(self) -> HealthStatus:
        """Check system health status.
        
        Returns:
            Current health status
        """
        try:
            with self.lock:
                health = HealthStatus()
                
                if not self.metrics_history:
                    health.status = "warning"
                    health.warnings.append("No metrics available")
                    return health
                
                latest_metrics = self.metrics_history[-1]
                
                # Check CPU usage
                if latest_metrics.cpu_usage > self.cpu_threshold:
                    health.cpu_ok = False
                    health.errors.append(f"High CPU usage: {latest_metrics.cpu_usage:.1f}%")
                
                # Check memory usage
                if latest_metrics.memory_usage > self.memory_threshold:\n                    health.memory_ok = False\n                    health.errors.append(f\"High memory usage: {latest_metrics.memory_usage:.1f}%\")\n                \n                # Check performance\n                if (latest_metrics.fps > 0 and latest_metrics.fps < self.min_fps):\n                    health.performance_ok = False\n                    health.warnings.append(f\"Low FPS: {latest_metrics.fps:.1f}\")\n                \n                if latest_metrics.step_time > self.max_step_time:\n                    health.performance_ok = False\n                    health.warnings.append(f\"Slow step time: {latest_metrics.step_time:.3f}s\")\n                \n                # Check for recent errors\n                if self.error_count > 0:\n                    health.warnings.append(f\"Recent errors: {self.error_count}\")\n                \n                # Determine overall status\n                if not health.cpu_ok or not health.memory_ok:\n                    health.status = \"critical\"\n                elif not health.performance_ok or health.warnings:\n                    health.status = \"warning\"\n                else:\n                    health.status = \"healthy\"\n                \n                # Store health status\n                self.health_history.append(health)\n                if len(self.health_history) > 100:\n                    self.health_history.pop(0)\n                \n                return health\n                \n        except Exception as e:\n            logger.error(f\"Health check failed: {str(e)}\")\n            self.error_count += 1\n            return HealthStatus(status=\"critical\", errors=[f\"Health check failed: {str(e)}\"])\n    \n    def get_performance_summary(self) -> Dict[str, Any]:\n        \"\"\"Get performance summary statistics.\n        \n        Returns:\n            Performance summary dictionary\n        \"\"\"\n        try:\n            with self.lock:\n                if not self.metrics_history:\n                    return {\"status\": \"no_data\", \"message\": \"No metrics available\"}\n                \n                recent_metrics = self.metrics_history[-10:]  # Last 10 entries\n                \n                avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)\n                avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)\n                avg_fps = sum(m.fps for m in recent_metrics if m.fps > 0) / max(1, sum(1 for m in recent_metrics if m.fps > 0))\n                avg_step_time = sum(m.step_time for m in recent_metrics if m.step_time > 0) / max(1, sum(1 for m in recent_metrics if m.step_time > 0))\n                \n                return {\n                    \"avg_cpu_usage\": avg_cpu,\n                    \"avg_memory_usage\": avg_memory,\n                    \"avg_fps\": avg_fps,\n                    \"avg_step_time\": avg_step_time,\n                    \"total_errors\": self.error_count,\n                    \"total_warnings\": self.warning_count,\n                    \"metrics_count\": len(self.metrics_history),\n                    \"health_status\": self.health_history[-1].status if self.health_history else \"unknown\"\n                }\n                \n        except Exception as e:\n            logger.error(f\"Failed to generate performance summary: {str(e)}\")\n            return {\"status\": \"error\", \"message\": str(e)}\n    \n    def _monitor_loop(self, interval: float) -> None:\n        \"\"\"Background monitoring loop.\"\"\"\n        logger.debug(\"Performance monitoring loop started\")\n        \n        while self.monitoring:\n            try:\n                # Record basic metrics\n                self.record_metrics()\n                \n                # Check health\n                health = self.check_health()\n                \n                # Log warnings/errors\n                if health.status == \"critical\":\n                    for error in health.errors:\n                        logger.error(f\"Critical health issue: {error}\")\n                elif health.status == \"warning\":\n                    for warning in health.warnings:\n                        logger.warning(f\"Performance warning: {warning}\")\n                \n                time.sleep(interval)\n                \n            except Exception as e:\n                logger.error(f\"Monitoring loop error: {str(e)}\")\n                self.error_count += 1\n                time.sleep(interval)\n        \n        logger.debug(\"Performance monitoring loop stopped\")\n    \n    def reset_counters(self) -> None:\n        \"\"\"Reset error and warning counters.\"\"\"\n        with self.lock:\n            self.error_count = 0\n            self.warning_count = 0\n            logger.info(\"Reset performance monitor counters\")\n\n\nclass SimulationHealthChecker:\n    \"\"\"Check simulation-specific health conditions.\"\"\"\n    \n    def __init__(self) -> None:\n        \"\"\"Initialize health checker.\"\"\"\n        self.checks: List[Callable[[], bool]] = []\n        self.check_names: List[str] = []\n        self.last_results: Dict[str, bool] = {}\n    \n    def add_check(self, name: str, check_func: Callable[[], bool]) -> None:\n        \"\"\"Add a health check function.\n        \n        Args:\n            name: Name of the check\n            check_func: Function that returns True if healthy\n        \"\"\"\n        self.checks.append(check_func)\n        self.check_names.append(name)\n        logger.debug(f\"Added health check: {name}\")\n    \n    def run_checks(self) -> Dict[str, bool]:\n        \"\"\"Run all health checks.\n        \n        Returns:\n            Dictionary of check results\n        \"\"\"\n        results = {}\n        \n        for name, check_func in zip(self.check_names, self.checks):\n            try:\n                result = check_func()\n                results[name] = bool(result)\n                \n                # Log status changes\n                if name in self.last_results and self.last_results[name] != result:\n                    if result:\n                        logger.info(f\"Health check '{name}' recovered\")\n                    else:\n                        logger.warning(f\"Health check '{name}' failed\")\n                        \n            except Exception as e:\n                logger.error(f\"Health check '{name}' threw exception: {str(e)}\")\n                results[name] = False\n        \n        self.last_results = results.copy()\n        return results\n    \n    def is_healthy(self) -> bool:\n        \"\"\"Check if all health checks pass.\n        \n        Returns:\n            True if all checks pass\n        \"\"\"\n        results = self.run_checks()\n        return all(results.values())\n\n\ndef create_default_health_checks(arena) -> SimulationHealthChecker:\n    \"\"\"Create default health checks for an arena.\n    \n    Args:\n        arena: Arena instance to monitor\n        \n    Returns:\n        Configured health checker\n    \"\"\"\n    checker = SimulationHealthChecker()\n    \n    # Check if arena has agents\n    checker.add_check(\n        \"has_agents\",\n        lambda: len(arena.agents) > 0\n    )\n    \n    # Check if any agents are alive\n    checker.add_check(\n        \"agents_alive\",\n        lambda: any(agent.state.alive for agent in arena.agents.values())\n    )\n    \n    # Check simulation step is progressing\n    last_step = [0]\n    def check_step_progress():\n        current_step = arena.current_step\n        if current_step > last_step[0]:\n            last_step[0] = current_step\n            return True\n        return current_step == 0  # Allow step 0\n    \n    checker.add_check(\"step_progress\", check_step_progress)\n    \n    # Check for reasonable step times\n    checker.add_check(\n        \"reasonable_step_times\",\n        lambda: not arena.step_times or max(arena.step_times[-10:]) < 5.0\n    )\n    \n    return checker