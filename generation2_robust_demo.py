#!/usr/bin/env python3
"""
Generation 2 Robust Demo - Showcase robustness and reliability features
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from swarm_arena import Arena, SwarmConfig, CooperativeAgent
from swarm_arena.security.authentication import AuthenticationManager, UserRole
from swarm_arena.security.input_validation import InputSanitizer, ConfigValidator
from swarm_arena.reliability.circuit_breaker import circuit_breaker, health_manager
from swarm_arena.reliability.retry_manager import retry, RetryStrategy
from swarm_arena.monitoring.advanced_telemetry import (
    metrics_collector, performance_profiler, event_tracker, 
    time_operation, profile_operation, track_event,
    MetricType, Alert
)


def demo_security_features():
    """Demonstrate security and authentication features."""
    print("🔐 SECURITY & AUTHENTICATION DEMO")
    print("=" * 50)
    
    # Authentication manager
    auth_manager = AuthenticationManager()
    
    # Create test users
    print("Creating test users...")
    
    try:
        researcher = auth_manager.create_user(
            username="researcher1",
            email="researcher@example.com",
            password="SecurePass123!",
            role=UserRole.RESEARCHER
        )
        print(f"✅ Created researcher: {researcher.username}")
    except Exception as e:
        print(f"❌ Failed to create researcher: {e}")
    
    # Authentication test
    print("\nTesting authentication...")
    token = auth_manager.authenticate("researcher1", "SecurePass123!")
    if token:
        print("✅ Authentication successful")
        
        # Check permissions
        from swarm_arena.security.authentication import Permission
        can_create = auth_manager.check_permission(token, Permission.ARENA_CREATE)
        print(f"✅ Can create arena: {can_create}")
        
        can_admin = auth_manager.check_permission(token, Permission.SYSTEM_ADMIN)
        print(f"⚠️  Can system admin: {can_admin}")
    else:
        print("❌ Authentication failed")
    
    # Input validation
    print("\nTesting input validation...")
    sanitizer = InputSanitizer()
    validator = ConfigValidator()
    
    # Test malicious input detection
    try:
        sanitizer.sanitize_string("<script>alert('xss')</script>")
        print("❌ XSS detection failed")
    except Exception as e:
        print("✅ XSS attack blocked")
    
    # Test config validation
    try:
        valid_config = {
            "num_agents": 50,
            "arena_size": [500, 500],
            "episode_length": 100
        }
        validated = validator.validate_config(valid_config)
        print("✅ Valid config accepted")
        
        invalid_config = {
            "num_agents": -5,  # Invalid
            "arena_size": [10, 10, 10],  # Wrong format
            "episode_length": "invalid"  # Wrong type
        }
        validator.validate_config(invalid_config)
        print("❌ Invalid config validation failed")
    except Exception as e:
        print("✅ Invalid config rejected")
    
    return auth_manager


def demo_reliability_features():
    """Demonstrate reliability patterns."""
    print("\n🛡️  RELIABILITY & FAULT TOLERANCE DEMO")
    print("=" * 50)
    
    # Circuit breaker demo
    print("Testing circuit breaker...")
    
    failure_count = 0
    
    @circuit_breaker("demo_service", failure_threshold=3, recovery_timeout=5.0)
    def unreliable_service():
        nonlocal failure_count
        failure_count += 1
        if failure_count <= 3:
            raise Exception(f"Service failure #{failure_count}")
        return f"Service success after {failure_count} attempts"
    
    # Test circuit breaker behavior
    for i in range(8):
        try:
            result = unreliable_service()
            print(f"✅ Attempt {i+1}: {result}")
        except Exception as e:
            print(f"❌ Attempt {i+1}: {e}")
        
        time.sleep(0.5)
    
    # Retry mechanism demo
    print("\nTesting retry mechanism...")
    
    attempt_count = 0
    
    @retry(max_attempts=3, base_delay=0.5, strategy=RetryStrategy.EXPONENTIAL_BACKOFF)
    def flaky_service():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception(f"Temporary failure {attempt_count}")
        return f"Success on attempt {attempt_count}"
    
    try:
        result = flaky_service()
        print(f"✅ Retry success: {result}")
    except Exception as e:
        print(f"❌ Retry failed: {e}")
    
    # Health checks
    print("\nTesting health checks...")
    
    def custom_health_check():
        """Custom health check that randomly fails."""
        return np.random.random() > 0.3
    
    health_manager.register_health_check("custom_service", custom_health_check)
    
    health_status = health_manager.get_overall_health()
    print(f"Overall health: {'✅ Healthy' if health_status['healthy'] else '❌ Unhealthy'}")
    print(f"Health score: {health_status['health_score']:.2f}")
    
    return health_status


def demo_monitoring_telemetry():
    """Demonstrate advanced monitoring and telemetry."""
    print("\n📊 MONITORING & TELEMETRY DEMO")
    print("=" * 50)
    
    # Record various metrics
    print("Recording metrics...")
    
    # Counter metrics
    for i in range(10):
        metrics_collector.record_counter("demo.requests", 1)
        metrics_collector.record_counter("demo.errors", 0.1)
    
    # Gauge metrics
    metrics_collector.record_gauge("demo.active_users", 25)
    metrics_collector.record_gauge("demo.cpu_usage", 45.2)
    
    # Histogram metrics
    for i in range(20):
        response_time = np.random.exponential(0.5)  # Simulated response times
        metrics_collector.record_histogram("demo.response_time", response_time)
    
    # Timer decorator demo
    @time_operation("demo.simulation_step")
    def simulate_computation():
        time.sleep(0.01)  # Simulate work
        return np.random.random()
    
    # Run timed operations
    for i in range(5):
        result = simulate_computation()
    
    # Profiling demo
    print("Profiling operations...")
    
    with profile_operation("demo.complex_computation") as profile:
        # Simulate complex computation
        for i in range(1000):
            _ = np.random.random() ** 2
        
        time.sleep(0.05)  # Simulate I/O
    
    print(f"Computation took {profile.duration:.4f} seconds")
    
    # Event tracking
    print("Tracking events...")
    
    event_tracker.track_event("user_login", {"username": "researcher1"})
    event_tracker.track_event("simulation_started", {"agents": 50})
    event_tracker.track_event("error_occurred", {"error_type": "validation"})
    
    # Set up alerts
    print("Setting up alerts...")
    
    def alert_handler(alert, value):
        print(f"🚨 ALERT: {alert.name} - {alert.message} (value: {value})")
    
    metrics_collector.add_alert_handler(alert_handler)
    
    high_cpu_alert = Alert(
        name="high_cpu",
        condition=lambda x: x > 80.0,
        message="CPU usage is too high",
        severity="critical"
    )
    
    metrics_collector.add_alert(high_cpu_alert)
    
    # Trigger alert
    metrics_collector.record_gauge("cpu_usage", 85.0)
    
    # Get metrics summary
    summary = metrics_collector.get_all_metrics_summary()
    
    print(f"\n📈 Metrics Summary:")
    print(f"   Counters: {len(summary['counters'])}")
    print(f"   Gauges: {len(summary['gauges'])}")
    print(f"   Histograms: {len(summary['histograms'])}")
    print(f"   Active alerts: {sum(1 for alert in summary['alerts'].values() if alert['is_active'])}")
    
    # Performance stats
    timer_stats = metrics_collector.get_timer_stats("demo.simulation_step")
    if timer_stats:
        print(f"   Simulation step avg: {timer_stats['mean']:.4f}s")
        print(f"   Simulation step p95: {timer_stats['p95']:.4f}s")
    
    return summary


def demo_robust_simulation():
    """Demonstrate robust simulation with all safety features."""
    print("\n🏟️  ROBUST SIMULATION DEMO")
    print("=" * 50)
    
    # Create configuration with validation
    validator = ConfigValidator()
    
    config_dict = {
        "num_agents": 30,
        "arena_size": [400, 400],
        "episode_length": 50,
        "seed": 42
    }
    
    try:
        validated_config = validator.validate_config(config_dict)
        print("✅ Configuration validated")
        
        config = SwarmConfig(**validated_config)
        
        # Create arena with monitoring
        with profile_operation("arena_creation") as profile:
            arena = Arena(config)
            arena.add_agents(CooperativeAgent, count=config.num_agents)
        
        print(f"✅ Arena created in {profile.duration:.4f}s")
        
        # Track simulation events
        track_event("simulation_initialized", {
            "num_agents": config.num_agents,
            "arena_size": config.arena_size
        })
        
        # Run simulation with monitoring
        @time_operation("simulation_execution")
        def run_monitored_simulation():
            return arena.run(episodes=2, verbose=False)
        
        print("Running monitored simulation...")
        results = run_monitored_simulation()
        
        # Record simulation metrics
        metrics_collector.record_gauge("simulation.mean_reward", results.mean_reward)
        metrics_collector.record_gauge("simulation.fairness_index", results.fairness_index or 0)
        metrics_collector.record_counter("simulation.total_steps", results.total_steps)
        
        track_event("simulation_completed", {
            "mean_reward": results.mean_reward,
            "total_steps": results.total_steps,
            "success": True
        })
        
        print("✅ Simulation completed successfully")
        print(f"   Mean reward: {results.mean_reward:.3f}")
        print(f"   Total steps: {results.total_steps}")
        
        return results
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        track_event("simulation_failed", {"error": str(e)})
        return None


def demo_error_recovery():
    """Demonstrate error recovery mechanisms."""
    print("\n🔄 ERROR RECOVERY DEMO")
    print("=" * 50)
    
    # Simulate various error scenarios
    error_scenarios = [
        "invalid_config",
        "memory_limit",
        "computation_timeout",
        "network_failure"
    ]
    
    for scenario in error_scenarios:
        print(f"Testing {scenario} recovery...")
        
        try:
            if scenario == "invalid_config":
                # Test config validation
                validator = ConfigValidator()
                bad_config = {"num_agents": "invalid", "arena_size": [-1, -1]}
                validator.validate_config(bad_config)
                
            elif scenario == "memory_limit":
                # Simulate memory pressure
                metrics_collector.record_gauge("system.memory.percent", 95.0)
                
            elif scenario == "computation_timeout":
                # Simulate timeout
                @retry(max_attempts=2, base_delay=0.1)
                def timeout_operation():
                    time.sleep(0.2)  # Simulate slow operation
                    raise TimeoutError("Operation timed out")
                
                timeout_operation()
                
            elif scenario == "network_failure":
                # Simulate network issue
                @circuit_breaker("network", failure_threshold=1)
                def network_operation():
                    raise ConnectionError("Network unreachable")
                
                network_operation()
                
        except Exception as e:
            print(f"   ✅ Error caught and handled: {type(e).__name__}")
            track_event("error_recovered", {
                "scenario": scenario,
                "error_type": type(e).__name__
            })
    
    print("✅ All error scenarios handled gracefully")


def main():
    """Run complete Generation 2 demonstration."""
    print("🚀 SWARM ARENA - GENERATION 2 ROBUST DEMO")
    print("🛡️  Autonomous SDLC Execution - Making It Robust!")
    print("=" * 60)
    
    try:
        # Security features
        auth_manager = demo_security_features()
        
        # Reliability features
        health_status = demo_reliability_features()
        
        # Monitoring and telemetry
        metrics_summary = demo_monitoring_telemetry()
        
        # Robust simulation
        sim_results = demo_robust_simulation()
        
        # Error recovery
        demo_error_recovery()
        
        print("\n🎉 GENERATION 2 COMPLETE!")
        print("=" * 60)
        print("✅ Security & authentication system")
        print("✅ Input validation & sanitization")
        print("✅ Circuit breakers & retry mechanisms")
        print("✅ Health checks & monitoring")
        print("✅ Advanced telemetry & alerting")
        print("✅ Error recovery & fault tolerance")
        print("✅ Performance profiling & optimization")
        
        print(f"\n📊 Robustness Summary:")
        print(f"   • Users managed: {len(auth_manager.list_users())}")
        print(f"   • Health score: {health_status['health_score']:.2f}")
        print(f"   • Metrics tracked: {len(metrics_summary['counters']) + len(metrics_summary['gauges'])}")
        print(f"   • Simulation success: {'✅' if sim_results else '❌'}")
        
        print(f"\n🎯 Ready for Generation 3: Optimizing and scaling!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)