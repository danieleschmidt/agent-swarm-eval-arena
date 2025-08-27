#!/usr/bin/env python3
"""
Enhanced monitoring and telemetry demonstration.
Shows real-time metrics collection and analysis capabilities.
"""

import sys
import os
import time
import json
from typing import Dict, List, Any
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simple_demo import SimpleArena, SimpleAgent


class TelemetryCollector:
    """Enhanced telemetry collection system."""
    
    def __init__(self, buffer_size: int = 1000):
        self.metrics = defaultdict(deque)
        self.buffer_size = buffer_size
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: float, timestamp: float = None) -> None:
        """Record a metric value."""
        if timestamp is None:
            timestamp = time.time()
        
        metric_buffer = self.metrics[name]
        metric_buffer.append((timestamp, value))
        
        if len(metric_buffer) > self.buffer_size:
            metric_buffer.popleft()
    
    def get_statistics(self, name: str, window_seconds: float = 60.0) -> Dict[str, float]:
        """Get statistics for a metric within time window."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        values = [value for timestamp, value in self.metrics[name] 
                 if timestamp >= cutoff_time]
        
        if not values:
            return {'count': 0, 'mean': 0, 'min': 0, 'max': 0, 'sum': 0}
        
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'sum': sum(values),
            'rate_per_second': len(values) / window_seconds
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive telemetry report."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        report = {
            'uptime_seconds': uptime,
            'collection_started': time.strftime('%Y-%m-%d %H:%M:%S', 
                                             time.localtime(self.start_time)),
            'metrics': {}
        }
        
        for metric_name in self.metrics:
            report['metrics'][metric_name] = self.get_statistics(metric_name)
        
        return report


class MonitoredAgent(SimpleAgent):
    """Agent with built-in monitoring capabilities."""
    
    def __init__(self, agent_id: int, x: float = 0, y: float = 0, telemetry: TelemetryCollector = None):
        super().__init__(agent_id, x, y)
        self.telemetry = telemetry
        self.decisions_made = 0
        self.total_distance_moved = 0.0
        self.last_position = (x, y)
    
    def act(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """Enhanced action with telemetry."""
        action = super().act(observation)
        
        self.decisions_made += 1
        if self.telemetry:
            self.telemetry.record_metric(f'agent_{self.id}_decisions', self.decisions_made)
            self.telemetry.record_metric(f'agent_{self.id}_energy', self.energy)
            
            # Record nearby agent count
            nearby_count = len(observation.get('nearby_agents', []))
            self.telemetry.record_metric(f'agent_{self.id}_nearby_agents', nearby_count)
        
        return action
    
    def step(self, action: Dict[str, float]) -> None:
        """Enhanced step with movement tracking."""
        old_x, old_y = self.x, self.y
        super().step(action)
        
        # Calculate distance moved
        distance = ((self.x - old_x) ** 2 + (self.y - old_y) ** 2) ** 0.5
        self.total_distance_moved += distance
        
        if self.telemetry:
            self.telemetry.record_metric(f'agent_{self.id}_distance_moved', distance)
            self.telemetry.record_metric(f'agent_{self.id}_total_distance', self.total_distance_moved)


class MonitoredArena(SimpleArena):
    """Arena with comprehensive monitoring."""
    
    def __init__(self, num_agents: int = 10, arena_size: tuple = (1000, 1000)):
        super().__init__(num_agents, arena_size)
        self.telemetry = TelemetryCollector()
        
        # Replace agents with monitored versions
        for agent_id in list(self.agents.keys()):
            old_agent = self.agents[agent_id]
            self.agents[agent_id] = MonitoredAgent(
                agent_id, old_agent.x, old_agent.y, self.telemetry
            )
        
        self.collision_count = 0
        self.total_energy_consumed = 0.0
    
    def step(self) -> Dict[str, Any]:
        """Enhanced step with monitoring."""
        step_start = time.time()
        
        # Record pre-step metrics
        total_energy_before = sum(agent.energy for agent in self.agents.values())
        
        result = super().step()
        
        # Record post-step metrics
        step_duration = time.time() - step_start
        total_energy_after = sum(agent.energy for agent in self.agents.values())
        energy_consumed = total_energy_before - total_energy_after
        
        self.telemetry.record_metric('step_duration', step_duration)
        self.telemetry.record_metric('total_energy', total_energy_after)
        self.telemetry.record_metric('energy_consumed_per_step', energy_consumed)
        self.telemetry.record_metric('agents_alive', 
                                   sum(1 for agent in self.agents.values() if agent.energy > 0))
        
        # Detect collisions (agents too close)
        collision_threshold = 10.0
        collisions_this_step = 0
        
        agent_positions = [(aid, agent.x, agent.y) for aid, agent in self.agents.items()]
        for i, (aid1, x1, y1) in enumerate(agent_positions):
            for j, (aid2, x2, y2) in enumerate(agent_positions[i+1:], i+1):
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if distance < collision_threshold:
                    collisions_this_step += 1
        
        self.collision_count += collisions_this_step
        self.telemetry.record_metric('collisions_per_step', collisions_this_step)
        self.telemetry.record_metric('total_collisions', self.collision_count)
        
        return result
    
    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Generate real-time dashboard data."""
        stats = {
            'simulation': {
                'step': self.step_count,
                'agents_total': len(self.agents),
                'agents_alive': sum(1 for agent in self.agents.values() if agent.energy > 0),
                'total_collisions': self.collision_count
            },
            'performance': self.telemetry.get_statistics('step_duration', 30),
            'energy': {
                'current_total': sum(agent.energy for agent in self.agents.values()),
                'consumption_rate': self.telemetry.get_statistics('energy_consumed_per_step', 60)
            },
            'movement': {
                'total_distance_all_agents': sum(agent.total_distance_moved 
                                               for agent in self.agents.values()),
                'average_distance_per_agent': sum(agent.total_distance_moved 
                                                for agent in self.agents.values()) / len(self.agents)
            }
        }
        
        return stats


def print_dashboard(arena: MonitoredArena, update_interval: float = 2.0):
    """Print real-time dashboard."""
    dashboard = arena.get_real_time_dashboard()
    
    print(f"\n📊 Real-Time Dashboard (Step {dashboard['simulation']['step']})")
    print("=" * 60)
    
    # Simulation metrics
    sim = dashboard['simulation']
    print(f"🤖 Agents: {sim['agents_alive']}/{sim['agents_total']} alive")
    print(f"💥 Collisions: {sim['total_collisions']}")
    
    # Performance metrics
    perf = dashboard['performance']
    print(f"⚡ Step Duration: {perf['mean']:.4f}s avg, {perf['rate_per_second']:.1f} steps/s")
    
    # Energy metrics
    energy = dashboard['energy']
    print(f"🔋 Total Energy: {energy['current_total']:.1f}")
    if energy['consumption_rate']['count'] > 0:
        print(f"📉 Consumption Rate: {energy['consumption_rate']['mean']:.2f}/step")
    
    # Movement metrics
    movement = dashboard['movement']
    print(f"🏃 Avg Distance/Agent: {movement['average_distance_per_agent']:.1f}")
    
    print("-" * 60)


def main():
    """Run enhanced monitoring demonstration."""
    print("🏟️  Swarm Arena - Enhanced Monitoring Demo")
    print("=" * 60)
    
    # Create monitored arena
    print("Initializing monitored arena with 30 agents...")
    arena = MonitoredArena(num_agents=30, arena_size=(1000, 1000))
    
    # Run with real-time monitoring
    print("\nRunning simulation with real-time monitoring...")
    max_steps = 200
    dashboard_interval = 50
    
    for step in range(max_steps):
        arena.step()
        
        # Show dashboard periodically
        if (step + 1) % dashboard_interval == 0:
            print_dashboard(arena)
    
    # Generate final comprehensive report
    print("\n📈 Final Telemetry Report")
    print("=" * 60)
    
    telemetry_report = arena.telemetry.generate_report()
    
    print(f"Simulation Duration: {telemetry_report['uptime_seconds']:.2f} seconds")
    print(f"Started: {telemetry_report['collection_started']}")
    
    # Key performance metrics
    key_metrics = ['step_duration', 'total_energy', 'agents_alive', 'total_collisions']
    for metric in key_metrics:
        if metric in telemetry_report['metrics']:
            stats = telemetry_report['metrics'][metric]
            print(f"\n{metric}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Min/Max: {stats['min']:.4f} / {stats['max']:.4f}")
    
    # Save detailed report
    report_filename = f"monitoring_report_{int(time.time())}.json"
    with open(report_filename, 'w') as f:
        json.dump(telemetry_report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {report_filename}")
    print("✅ Enhanced monitoring demo completed successfully!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)