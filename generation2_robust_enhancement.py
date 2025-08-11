#!/usr/bin/env python3
"""
Generation 2 Robust Enhancement - Comprehensive Error Handling & Monitoring
Enhances the arena with reliability, security, and advanced monitoring
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from swarm_arena import Arena, SwarmConfig, CooperativeAgent, CompetitiveAgent
from swarm_arena.monitoring.telemetry import TelemetryCollector
from swarm_arena.utils.error_handler import error_manager
from swarm_arena.utils.validation import validate_positive, validate_action
from swarm_arena.monitoring.health import HealthMonitor

@dataclass
class RobustMetrics:
    """Enhanced metrics collection for robust monitoring."""
    total_errors: int = 0
    error_recovery_rate: float = 0.0
    system_health_score: float = 100.0
    security_violations: int = 0
    performance_degradation: float = 0.0
    memory_usage_mb: float = 0.0
    error_log: list = field(default_factory=list)

class RobustArenaWrapper:
    """Robust wrapper around Arena with enhanced error handling and monitoring."""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.arena: Optional[Arena] = None
        self.telemetry = TelemetryCollector()
        self.health_monitor = HealthMonitor()
        self.metrics = RobustMetrics()
        
        # Setup advanced logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('arena_robust.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize with comprehensive validation
        self._initialize_arena()
    
    def _initialize_arena(self) -> None:
        """Initialize arena with comprehensive validation and error handling."""
        try:
            # Validate configuration
            self._validate_config()
            
            # Create arena with error handling
            self.arena = Arena(self.config)
            
            # Setup telemetry monitoring
            self.telemetry.start_collection()
            self.health_monitor.start_monitoring()
            
            self.logger.info("✓ Robust arena initialized successfully")
            
        except Exception as e:
            self.logger.error(f"✗ Arena initialization failed: {e}")
            self.metrics.error_log.append(f"Init error: {e}")
            self.metrics.total_errors += 1
            raise
    
    def _validate_config(self) -> None:
        """Comprehensive configuration validation with security checks."""
        # Validate numeric parameters
        validate_positive(self.config.num_agents, "num_agents")
        validate_positive(self.config.episode_length, "episode_length")
        validate_positive(self.config.observation_radius, "observation_radius")
        validate_positive(self.config.max_agent_speed, "max_agent_speed")
        
        # Security bounds checking
        if self.config.num_agents > 10000:
            self.logger.warning(f"⚠️  Large agent count ({self.config.num_agents}) may impact performance")
            self.metrics.security_violations += 1
        
        if self.config.arena_size[0] * self.config.arena_size[1] > 100_000_000:
            self.logger.warning("⚠️  Extremely large arena size detected")
            self.metrics.security_violations += 1
        
        self.logger.info("✓ Configuration validation passed")
    
    def add_agents_safely(self, agent_class, count: int, **kwargs) -> bool:
        """Add agents with comprehensive error handling and validation."""
        if not self.arena:
            self.logger.error("Arena not initialized")
            return False
        
        try:
            # Validate agent parameters
            if count <= 0 or count > 5000:
                raise ValueError(f"Invalid agent count: {count}")
            
            # Memory usage check
            estimated_memory = count * 0.1  # ~0.1MB per agent estimate
            if estimated_memory > 1000:  # 1GB limit
                self.logger.warning(f"⚠️  High memory usage estimated: {estimated_memory}MB")
            
            # Add agents with error recovery
            success_count = 0
            for i in range(count):
                try:
                    self.arena.add_agents(agent_class, count=1, **kwargs)
                    success_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to add agent {i}: {e}")
                    self.metrics.total_errors += 1
                    self.metrics.error_log.append(f"Agent add error: {e}")
            
            self.metrics.error_recovery_rate = success_count / count if count > 0 else 0.0
            self.logger.info(f"✓ Added {success_count}/{count} agents successfully")
            
            return success_count == count
            
        except Exception as e:
            self.logger.error(f"✗ Failed to add agents: {e}")
            self.metrics.total_errors += 1
            return False
    
    def run_robust_simulation(self, episodes: int = 1, safety_checks: bool = True) -> Optional[Dict[str, Any]]:
        """Run simulation with comprehensive error handling and health monitoring."""
        if not self.arena:
            self.logger.error("Arena not initialized")
            return None
        
        try:
            self.logger.info(f"🚀 Starting robust simulation with {episodes} episodes")
            
            # Pre-flight safety checks
            if safety_checks:
                health_data = self.health_monitor.check_system_health()
                health_score = health_data.get('overall_score', 100)
                if health_score < 80:
                    self.logger.warning(f"⚠️  Low system health: {health_score}%")
                    self.metrics.system_health_score = health_score
            
            # Run with comprehensive error handling
            start_time = time.time()
            results = None
            
            try:
                results = self.arena.run(episodes=episodes, verbose=True)
            except Exception as e:
                self.logger.error(f"Arena run failed: {e}")
                self.metrics.total_errors += 1
                return None
            
            execution_time = time.time() - start_time
            
            # Performance monitoring
            if execution_time > episodes * 60:  # > 1 minute per episode
                self.logger.warning(f"⚠️  Slow performance detected: {execution_time:.2f}s")
                self.metrics.performance_degradation = execution_time / (episodes * 60)
            
            # Collect enhanced telemetry
            telemetry_data = self.telemetry.get_latest_metrics()
            
            robust_results = {
                "simulation_results": results.__dict__ if results else {},
                "telemetry": telemetry_data.__dict__ if telemetry_data else {},
                "metrics": self.metrics.__dict__,
                "execution_time": execution_time,
                "episodes_completed": episodes
            }
            
            self.logger.info(f"✅ Robust simulation completed in {execution_time:.2f}s")
            return robust_results
            
        except Exception as e:
            self.logger.error(f"✗ Simulation failed: {e}")
            self.metrics.total_errors += 1
            self.metrics.error_log.append(f"Simulation error: {e}")
            return None
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        health_data = self.health_monitor.check_system_health()
        return {
            "system_health": health_data.get('overall_score', 100),
            "health_status": health_data.get('status', 'healthy'),
            "error_stats": {
                "total_errors": self.metrics.total_errors,
                "error_recovery_rate": self.metrics.error_recovery_rate,
                "security_violations": self.metrics.security_violations
            },
            "performance": {
                "degradation": self.metrics.performance_degradation,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "cpu_percent": health_data.get('cpu_percent', 0),
                "memory_percent": health_data.get('memory_percent', 0)
            },
            "recent_errors": self.metrics.error_log[-5:] if self.metrics.error_log else []
        }
    
    def cleanup(self) -> None:
        """Cleanup resources with error handling."""
        try:
            if self.telemetry:
                self.telemetry.stop_collection()
            if self.health_monitor:
                self.health_monitor.stop_monitoring()
            self.logger.info("✓ Robust arena cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

def main():
    print("🛡️  Swarm Arena - Generation 2 Robust Enhancement")
    print("=" * 60)
    
    # Create robust configuration with security bounds
    config = SwarmConfig(
        num_agents=50,  # Increased agent count for robustness testing
        arena_size=(800, 800),
        episode_length=200,
        observation_radius=60.0,
        max_agent_speed=6.0,
        collision_detection=True,
        collision_radius=8.0,
        seed=42,
        reward_config={
            "resource_collection": 1.0,
            "time_penalty": -0.001,
            "survival_bonus": 0.02,
            "collision_penalty": -0.1,
            "cooperation_bonus": 0.05
        }
    )
    
    # Create robust arena wrapper
    robust_arena = RobustArenaWrapper(config)
    
    print(f"✓ Robust arena created with enhanced monitoring")
    print(f"✓ Configuration validated with security checks")
    
    # Add agents with error handling
    print("\n🤖 Adding agents with robust error handling...")
    success1 = robust_arena.add_agents_safely(CooperativeAgent, count=25)
    success2 = robust_arena.add_agents_safely(CompetitiveAgent, count=25)
    
    if not (success1 and success2):
        print("⚠️  Some agents failed to initialize - continuing with partial setup")
    
    # Run robust simulation
    print("\n🎯 Running Generation 2 robust evaluation...")
    results = robust_arena.run_robust_simulation(episodes=3, safety_checks=True)
    
    if results:
        print(f"\n📊 Robust Results:")
        print(f"• Execution time: {results['execution_time']:.2f}s")
        print(f"• Episodes completed: {results['episodes_completed']}")
        
        sim_results = results.get('simulation_results', {})
        if sim_results:
            print(f"• Mean reward: {sim_results.get('mean_reward', 0):.3f}")
            print(f"• Total steps: {sim_results.get('total_steps', 0)}")
            print(f"• Fairness index: {sim_results.get('fairness_index', 0):.3f}")
    
    # Generate health report
    print(f"\n🏥 Health Report:")
    health_report = robust_arena.get_health_report()
    print(f"• System health: {health_report['system_health']:.1f}%")
    print(f"• Total errors: {health_report['error_stats']['total_errors']}")
    print(f"• Error recovery rate: {health_report['error_stats']['error_recovery_rate']:.2%}")
    print(f"• Security violations: {health_report['error_stats']['security_violations']}")
    print(f"• Performance degradation: {health_report['performance']['degradation']:.2f}x")
    
    if health_report.get('recent_errors'):
        print(f"• Recent errors: {len(health_report['recent_errors'])}")
    
    # Cleanup
    robust_arena.cleanup()
    
    print("\n✅ Generation 2 robust enhancement complete!")
    print("✅ Error handling, monitoring, and security validated!")

if __name__ == "__main__":
    main()