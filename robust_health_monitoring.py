#!/usr/bin/env python3
"""
Robust health monitoring and self-healing capabilities.
Implements circuit breakers, retry logic, and adaptive recovery systems.
"""

import sys
import os
import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from enum import Enum
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enhanced_monitoring_demo import MonitoredArena, MonitoredAgent, TelemetryCollector


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    FAILED = "failed"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests rejected
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitBreakerState.CLOSED
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception("Circuit breaker OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (self.last_failure_time is not None and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, HealthStatus] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_health_check(self, name: str, check_func: Callable, 
                        enable_circuit_breaker: bool = True) -> None:
        """Add a health check function."""
        self.health_checks[name] = check_func
        self.health_status[name] = HealthStatus.HEALTHY
        
        if enable_circuit_breaker:
            self.circuit_breakers[name] = CircuitBreaker()
    
    def start_monitoring(self) -> None:
        """Start health monitoring in background thread."""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                self._run_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
    
    def _run_health_checks(self) -> None:
        """Execute all health checks."""
        for name, check_func in self.health_checks.items():
            try:
                if name in self.circuit_breakers:
                    # Use circuit breaker
                    result = self.circuit_breakers[name].call(check_func)
                else:
                    result = check_func()
                
                # Interpret result
                if isinstance(result, bool):
                    status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                elif isinstance(result, HealthStatus):
                    status = result
                elif isinstance(result, dict):
                    # Allow detailed health responses
                    status = HealthStatus(result.get('status', 'healthy'))
                else:
                    status = HealthStatus.HEALTHY
                
                self._update_health_status(name, status)
                
            except Exception as e:
                self.logger.warning(f"Health check '{name}' failed: {e}")
                self._update_health_status(name, HealthStatus.FAILED)
    
    def _update_health_status(self, name: str, status: HealthStatus) -> None:
        """Update health status and trigger alerts if needed."""
        previous_status = self.health_status.get(name, HealthStatus.HEALTHY)
        self.health_status[name] = status
        
        # Record in history
        self.health_history[name].append({
            'timestamp': time.time(),
            'status': status.value
        })
        
        # Trigger alert on status change
        if status != previous_status:
            self._trigger_alert(name, previous_status, status)
    
    def _trigger_alert(self, component: str, old_status: HealthStatus, 
                      new_status: HealthStatus) -> None:
        """Trigger health status change alert."""
        alert = {
            'timestamp': time.time(),
            'component': component,
            'old_status': old_status.value,
            'new_status': new_status.value,
            'severity': self._get_alert_severity(new_status)
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"Health alert: {component} {old_status.value} -> {new_status.value}")
    
    def _get_alert_severity(self, status: HealthStatus) -> str:
        """Get alert severity level."""
        severity_map = {
            HealthStatus.HEALTHY: 'info',
            HealthStatus.DEGRADED: 'warning', 
            HealthStatus.CRITICAL: 'error',
            HealthStatus.FAILED: 'critical'
        }
        return severity_map.get(status, 'info')
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.health_status:
            return {'status': 'unknown', 'components': 0}
        
        status_counts = defaultdict(int)
        for status in self.health_status.values():
            status_counts[status.value] += 1
        
        # Determine overall status
        if status_counts['failed'] > 0:
            overall_status = 'failed'
        elif status_counts['critical'] > 0:
            overall_status = 'critical'
        elif status_counts['degraded'] > 0:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        return {
            'status': overall_status,
            'components': len(self.health_status),
            'healthy': status_counts['healthy'],
            'degraded': status_counts['degraded'],
            'critical': status_counts['critical'],
            'failed': status_counts['failed'],
            'recent_alerts': len([a for a in self.alerts 
                                if time.time() - a['timestamp'] < 300])  # Last 5 minutes
        }


class SelfHealingArena(MonitoredArena):
    """Arena with self-healing capabilities."""
    
    def __init__(self, num_agents: int = 10, arena_size: tuple = (1000, 1000)):
        super().__init__(num_agents, arena_size)
        self.health_monitor = HealthMonitor(check_interval=3.0)
        self.recovery_strategies: Dict[str, Callable] = {}
        self.auto_recovery_enabled = True
        
        # Setup health checks
        self._setup_health_checks()
        self.health_monitor.start_monitoring()
        
        # Initialize recovery metrics
        self.recovery_attempts = 0
        self.successful_recoveries = 0
    
    def _setup_health_checks(self) -> None:
        """Setup health checks for arena components."""
        
        def check_agent_health():
            """Check if agents are functioning properly."""
            if not self.agents:
                return HealthStatus.FAILED
            
            alive_agents = sum(1 for agent in self.agents.values() if agent.energy > 0)
            alive_ratio = alive_agents / len(self.agents)
            
            if alive_ratio > 0.8:
                return HealthStatus.HEALTHY
            elif alive_ratio > 0.3:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.CRITICAL
        
        def check_simulation_performance():
            """Check simulation performance metrics."""
            step_duration_stats = self.telemetry.get_statistics('step_duration', 30)
            
            if step_duration_stats['count'] == 0:
                return HealthStatus.HEALTHY
            
            avg_duration = step_duration_stats['mean']
            if avg_duration < 0.01:  # Less than 10ms per step
                return HealthStatus.HEALTHY
            elif avg_duration < 0.05:  # Less than 50ms per step
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.CRITICAL
        
        def check_collision_rate():
            """Check for excessive collision rates."""
            collision_stats = self.telemetry.get_statistics('collisions_per_step', 60)
            
            if collision_stats['count'] == 0:
                return HealthStatus.HEALTHY
            
            avg_collisions = collision_stats['mean']
            if avg_collisions < 2:
                return HealthStatus.HEALTHY
            elif avg_collisions < 5:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.CRITICAL
        
        # Register health checks
        self.health_monitor.add_health_check('agents', check_agent_health)
        self.health_monitor.add_health_check('performance', check_simulation_performance)
        self.health_monitor.add_health_check('collisions', check_collision_rate)
        
        # Register recovery strategies
        self.recovery_strategies['agents'] = self._recover_agents
        self.recovery_strategies['performance'] = self._recover_performance
        self.recovery_strategies['collisions'] = self._recover_collisions
    
    def _recover_agents(self) -> bool:
        """Attempt to recover unhealthy agents."""
        try:
            self.recovery_attempts += 1
            
            # Restore energy to low-energy agents
            recovered_count = 0
            for agent in self.agents.values():
                if agent.energy <= 0:
                    agent.energy = 50.0  # Partial energy restoration
                    recovered_count += 1
            
            if recovered_count > 0:
                self.telemetry.record_metric('agents_recovered', recovered_count)
                self.successful_recoveries += 1
                return True
            
        except Exception as e:
            self.health_monitor.logger.error(f"Agent recovery failed: {e}")
        
        return False
    
    def _recover_performance(self) -> bool:
        """Attempt to recover performance issues."""
        try:
            self.recovery_attempts += 1
            
            # Reduce agent count temporarily to improve performance
            if len(self.agents) > 10:
                agents_to_remove = list(self.agents.keys())[-5:]  # Remove last 5 agents
                for agent_id in agents_to_remove:
                    del self.agents[agent_id]
                
                self.telemetry.record_metric('agents_removed_for_performance', len(agents_to_remove))
                self.successful_recoveries += 1
                return True
                
        except Exception as e:
            self.health_monitor.logger.error(f"Performance recovery failed: {e}")
        
        return False
    
    def _recover_collisions(self) -> bool:
        """Attempt to recover from collision issues."""
        try:
            self.recovery_attempts += 1
            
            # Spread agents out more to reduce collisions
            import random
            for agent in self.agents.values():
                agent.x = random.uniform(0, self.width)
                agent.y = random.uniform(0, self.height)
            
            self.telemetry.record_metric('collision_recovery_spreads', 1)
            self.successful_recoveries += 1
            return True
            
        except Exception as e:
            self.health_monitor.logger.error(f"Collision recovery failed: {e}")
        
        return False
    
    def step(self) -> Dict[str, Any]:
        """Enhanced step with self-healing capabilities."""
        result = super().step()
        
        # Check for health issues and attempt recovery
        if self.auto_recovery_enabled:
            overall_health = self.health_monitor.get_overall_health()
            
            if overall_health['status'] in ['critical', 'failed']:
                self._attempt_recovery()
        
        return result
    
    def _attempt_recovery(self) -> None:
        """Attempt to recover from health issues."""
        for component, status in self.health_monitor.health_status.items():
            if status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                if component in self.recovery_strategies:
                    recovery_func = self.recovery_strategies[component]
                    try:
                        if recovery_func():
                            self.health_monitor.logger.info(f"Successfully recovered {component}")
                    except Exception as e:
                        self.health_monitor.logger.error(f"Recovery attempt failed for {component}: {e}")
    
    def get_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive health dashboard."""
        overall_health = self.health_monitor.get_overall_health()
        
        dashboard = {
            'overall_health': overall_health,
            'component_health': {name: status.value 
                               for name, status in self.health_monitor.health_status.items()},
            'recent_alerts': self.health_monitor.alerts[-10:],  # Last 10 alerts
            'recovery_stats': {
                'attempts': self.recovery_attempts,
                'successful': self.successful_recoveries,
                'success_rate': (self.successful_recoveries / self.recovery_attempts 
                               if self.recovery_attempts > 0 else 0.0)
            },
            'circuit_breakers': {
                name: breaker.state.value 
                for name, breaker in self.health_monitor.circuit_breakers.items()
            }
        }
        
        return dashboard
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'health_monitor'):
            self.health_monitor.stop_monitoring()


def print_health_dashboard(arena: SelfHealingArena):
    """Print comprehensive health dashboard."""
    dashboard = arena.get_health_dashboard()
    
    print(f"\n🏥 Health Dashboard (Step {arena.step_count})")
    print("=" * 70)
    
    # Overall health
    overall = dashboard['overall_health']
    health_emoji = {
        'healthy': '✅',
        'degraded': '⚠️', 
        'critical': '🚨',
        'failed': '❌'
    }
    
    print(f"{health_emoji[overall['status']]} Overall Status: {overall['status'].upper()}")
    print(f"📊 Components: {overall['components']} total")
    print(f"   Healthy: {overall['healthy']}, Degraded: {overall['degraded']}")
    print(f"   Critical: {overall['critical']}, Failed: {overall['failed']}")
    
    # Component health
    print(f"\n🔧 Component Health:")
    for component, status in dashboard['component_health'].items():
        emoji = health_emoji.get(status, '❓')
        print(f"  {emoji} {component}: {status}")
    
    # Recovery stats
    recovery = dashboard['recovery_stats']
    print(f"\n🔄 Recovery Statistics:")
    print(f"  Attempts: {recovery['attempts']}")
    print(f"  Successful: {recovery['successful']}")
    print(f"  Success Rate: {recovery['success_rate']:.1%}")
    
    # Recent alerts
    if dashboard['recent_alerts']:
        print(f"\n🚨 Recent Alerts ({len(dashboard['recent_alerts'])}):")
        for alert in dashboard['recent_alerts'][-3:]:  # Show last 3
            timestamp = time.strftime('%H:%M:%S', time.localtime(alert['timestamp']))
            print(f"  {timestamp} {alert['component']}: {alert['old_status']} -> {alert['new_status']}")
    
    print("-" * 70)


def main():
    """Run robust health monitoring demonstration."""
    print("🏟️  Swarm Arena - Robust Health Monitoring Demo")
    print("=" * 70)
    
    # Create self-healing arena
    print("Initializing self-healing arena with 25 agents...")
    arena = SelfHealingArena(num_agents=25, arena_size=(800, 800))
    
    print("🏥 Health monitoring system started")
    print("🔄 Auto-recovery enabled")
    
    # Run simulation with health monitoring
    max_steps = 300
    dashboard_interval = 50
    
    for step in range(max_steps):
        arena.step()
        
        # Show health dashboard periodically
        if (step + 1) % dashboard_interval == 0:
            print_health_dashboard(arena)
        
        # Simulate some stress conditions
        if step == 100:
            # Drain agent energy to trigger health issues
            print("\n⚡ Simulating energy crisis...")
            for agent in list(arena.agents.values())[:15]:
                agent.energy = -10
        
        if step == 200:
            # Add more agents to stress performance
            print("\n📈 Adding stress load (more agents)...")
            for i in range(50, 65):
                arena.agents[i] = MonitoredAgent(i, 400, 400, arena.telemetry)
    
    # Final health report
    print("\n📋 Final Health Assessment")
    print("=" * 70)
    
    final_dashboard = arena.get_health_dashboard()
    overall = final_dashboard['overall_health']
    recovery = final_dashboard['recovery_stats']
    
    print(f"Final Status: {overall['status'].upper()}")
    print(f"Total Recovery Attempts: {recovery['attempts']}")
    print(f"Successful Recoveries: {recovery['successful']}")
    print(f"Recovery Success Rate: {recovery['success_rate']:.1%}")
    print(f"Total Alerts Generated: {len(arena.health_monitor.alerts)}")
    
    # Save health report
    report_filename = f"health_report_{int(time.time())}.json"
    with open(report_filename, 'w') as f:
        json.dump(final_dashboard, f, indent=2, default=str)
    
    print(f"\n💾 Health report saved to: {report_filename}")
    
    # Cleanup
    arena.health_monitor.stop_monitoring()
    print("✅ Robust health monitoring demo completed successfully!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)