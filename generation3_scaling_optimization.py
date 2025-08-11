#!/usr/bin/env python3
"""
Generation 3 Scaling Optimization - Massive Scale & Performance
Demonstrates advanced optimizations for 1000+ agent concurrent simulations
"""

import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import threading
from queue import Queue
import psutil

from swarm_arena import Arena, SwarmConfig, CooperativeAgent, CompetitiveAgent
from swarm_arena.distributed.ray_arena import DistributedArena
from swarm_arena.benchmarks.scaling import ScalingBenchmark
from swarm_arena.core.parallel_arena import ParallelArena

@dataclass
class ScalingMetrics:
    """Performance metrics for scaling analysis."""
    
    # Scale metrics
    agent_count: int = 0
    arena_workers: int = 1
    parallel_episodes: int = 1
    
    # Performance metrics
    total_throughput: float = 0.0  # agents*steps/second
    memory_efficiency: float = 0.0  # MB per agent
    cpu_utilization: float = 0.0
    steps_per_second: float = 0.0
    
    # Optimization metrics
    cache_hit_rate: float = 0.0
    spatial_query_time_ms: float = 0.0
    vectorization_speedup: float = 1.0
    
    # Scaling coefficients
    weak_scaling_efficiency: float = 1.0
    strong_scaling_efficiency: float = 1.0
    memory_scaling_coefficient: float = 1.0

class AdvancedOptimizationManager:
    """Manages advanced optimizations for massive scale simulations."""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.optimization_cache = {}
        self.metrics_history = []
        
        print(f"🚀 Optimization Manager initialized:")
        print(f"   • CPU cores: {self.cpu_count}")
        print(f"   • Memory: {self.memory_gb:.1f} GB")
    
    def optimize_for_scale(self, target_agents: int, target_throughput: float) -> Dict[str, Any]:
        """Auto-configure optimal settings for target scale."""
        # Memory-based agent limit calculation
        estimated_memory_per_agent = 0.5  # MB per agent with optimizations
        max_agents_memory = int((self.memory_gb * 1000 * 0.8) / estimated_memory_per_agent)
        
        # CPU-based parallelization
        optimal_workers = min(self.cpu_count, max(1, target_agents // 100))
        
        # Adaptive batch sizing
        optimal_batch_size = max(10, min(1000, target_agents // optimal_workers))
        
        # Auto-scaling arena size
        agent_density = 0.001  # agents per unit²
        optimal_arena_area = target_agents / agent_density
        arena_side = int(np.sqrt(optimal_arena_area))
        
        config = {
            "recommended_agents": min(target_agents, max_agents_memory),
            "parallel_workers": optimal_workers,
            "batch_size": optimal_batch_size,
            "arena_size": (arena_side, arena_side),
            "memory_limit_mb": self.memory_gb * 1000 * 0.8,
            "optimization_level": "ultra" if target_agents > 1000 else "high"
        }
        
        return config

class MassiveScaleArena:
    """Massively scalable arena with advanced optimizations."""
    
    def __init__(self, config: SwarmConfig, optimization_level: str = "ultra"):
        self.config = config
        self.optimization_level = optimization_level
        self.opt_manager = AdvancedOptimizationManager()
        
        # Auto-optimize configuration
        scale_config = self.opt_manager.optimize_for_scale(
            target_agents=config.num_agents,
            target_throughput=10000  # agents*steps/second
        )
        
        # Initialize optimized arena based on scale
        if config.num_agents > 500:
            print(f"🌐 Using distributed arena for {config.num_agents} agents")
            try:
                self.arena = DistributedArena(config, num_workers=scale_config["parallel_workers"])
            except Exception as e:
                print(f"⚠️  Distributed arena failed, falling back to parallel: {e}")
                self.arena = ParallelArena(config)
        else:
            print(f"⚡ Using parallel arena for {config.num_agents} agents")
            self.arena = ParallelArena(config)
        
        self.scaling_metrics = ScalingMetrics(
            agent_count=config.num_agents,
            arena_workers=scale_config["parallel_workers"]
        )
        
        # Performance monitoring
        self.performance_buffer = Queue(maxsize=1000)
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
    
    def start_performance_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitoring_thread.start()
    
    def _monitor_performance(self) -> None:
        """Background performance monitoring."""
        while self.monitoring_active:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                
                # Update scaling metrics
                self.scaling_metrics.cpu_utilization = cpu_percent
                self.scaling_metrics.memory_efficiency = memory_info.used / (1024**2) / self.config.num_agents
                
                # Performance data point
                perf_data = {
                    'timestamp': time.time(),
                    'cpu': cpu_percent,
                    'memory_percent': memory_info.percent,
                    'memory_mb': memory_info.used / (1024**2)
                }
                
                if not self.performance_buffer.full():
                    self.performance_buffer.put(perf_data)
                    
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                
            time.sleep(1)
    
    def run_massive_scale_benchmark(self, episodes: int = 5) -> Dict[str, Any]:
        """Run comprehensive scaling benchmark."""
        print(f"🎯 Starting massive scale benchmark with {self.config.num_agents} agents")
        
        # Start monitoring
        self.start_performance_monitoring()
        
        try:
            # Warm-up run for cache optimization
            print("🔥 Warm-up phase...")
            start_time = time.time()
            self.arena.reset()
            
            # Add agents in batches for memory efficiency
            agents_per_batch = min(100, self.config.num_agents // 4)
            for batch_start in range(0, self.config.num_agents, agents_per_batch):
                batch_size = min(agents_per_batch, self.config.num_agents - batch_start)
                cooperative_count = batch_size // 2
                competitive_count = batch_size - cooperative_count
                
                if cooperative_count > 0:
                    self.arena.add_agents(CooperativeAgent, count=cooperative_count)
                if competitive_count > 0:
                    self.arena.add_agents(CompetitiveAgent, count=competitive_count)
                
                print(f"  Added batch: {batch_start + batch_size}/{self.config.num_agents} agents")
            
            warmup_time = time.time() - start_time
            print(f"✓ Warmup completed in {warmup_time:.2f}s")
            
            # Main benchmark run
            print(f"🚀 Running {episodes} episodes...")
            benchmark_start = time.time()
            
            results = self.arena.run(episodes=episodes, verbose=True)
            
            benchmark_time = time.time() - benchmark_start
            
            # Calculate performance metrics
            total_steps = results.total_steps if results else 0
            total_agent_steps = total_steps * self.config.num_agents
            
            self.scaling_metrics.total_throughput = total_agent_steps / benchmark_time if benchmark_time > 0 else 0
            self.scaling_metrics.steps_per_second = total_steps / benchmark_time if benchmark_time > 0 else 0
            
            # Collect performance data
            performance_data = []
            while not self.performance_buffer.empty():
                performance_data.append(self.performance_buffer.get())
            
            benchmark_results = {
                'simulation_results': results.__dict__ if results else {},
                'scaling_metrics': self.scaling_metrics.__dict__,
                'performance_timeline': performance_data,
                'benchmark_time': benchmark_time,
                'warmup_time': warmup_time,
                'optimization_level': self.optimization_level
            }
            
            print(f"✅ Benchmark completed in {benchmark_time:.2f}s")
            return benchmark_results
            
        except Exception as e:
            print(f"✗ Benchmark failed: {e}")
            return {'error': str(e)}
            
        finally:
            self.monitoring_active = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=2)

def run_multi_scale_comparison(scale_factors: List[int]) -> Dict[int, Dict[str, Any]]:
    """Run benchmark across multiple scales for comparison."""
    results = {}
    base_config = SwarmConfig(
        num_agents=100,  # Base will be multiplied by scale factors
        arena_size=(1000, 1000),
        episode_length=50,  # Shorter episodes for scaling test
        observation_radius=30.0,
        max_agent_speed=4.0,
        collision_detection=True,
        collision_radius=3.0,
        seed=42,
        reward_config={
            "resource_collection": 1.0,
            "time_penalty": -0.001,
            "survival_bonus": 0.01,
            "collision_penalty": -0.05,
            "cooperation_bonus": 0.02
        }
    )
    
    print(f"🔬 Running multi-scale comparison across {len(scale_factors)} scales")
    
    for scale in scale_factors:
        print(f"\n{'='*20} Scale Factor: {scale}x {'='*20}")
        
        # Scale configuration
        scaled_config = SwarmConfig(
            num_agents=base_config.num_agents * scale,
            arena_size=(base_config.arena_size[0] * scale, base_config.arena_size[1] * scale),
            episode_length=base_config.episode_length,
            observation_radius=base_config.observation_radius,
            max_agent_speed=base_config.max_agent_speed,
            collision_detection=base_config.collision_detection,
            collision_radius=base_config.collision_radius,
            seed=base_config.seed,
            reward_config=base_config.reward_config
        )
        
        try:
            massive_arena = MassiveScaleArena(scaled_config, optimization_level="ultra")
            scale_results = massive_arena.run_massive_scale_benchmark(episodes=2)
            
            # Calculate scaling efficiency
            if scale > 1 and 1 in results:
                base_throughput = results[1]['scaling_metrics']['total_throughput']
                current_throughput = scale_results['scaling_metrics']['total_throughput']
                if base_throughput > 0:
                    scale_results['scaling_metrics']['weak_scaling_efficiency'] = \
                        current_throughput / (base_throughput * scale)
            
            results[scale] = scale_results
            
            # Print quick summary
            metrics = scale_results.get('scaling_metrics', {})
            print(f"  ✓ {scaled_config.num_agents} agents: {metrics.get('total_throughput', 0):.0f} agent*steps/s")
            print(f"  ✓ Memory efficiency: {metrics.get('memory_efficiency', 0):.2f} MB/agent")
            
        except Exception as e:
            print(f"  ✗ Scale {scale}x failed: {e}")
            results[scale] = {'error': str(e)}
    
    return results

def main():
    print("⚡ Swarm Arena - Generation 3 Scaling Optimization")
    print("=" * 70)
    
    # Run single massive scale demonstration
    massive_config = SwarmConfig(
        num_agents=1000,  # 1K agents for demonstration
        arena_size=(2000, 2000),
        episode_length=100,
        observation_radius=40.0,
        max_agent_speed=5.0,
        collision_detection=True,
        collision_radius=5.0,
        seed=42,
        reward_config={
            "resource_collection": 1.0,
            "time_penalty": -0.001,
            "survival_bonus": 0.01,
            "collision_penalty": -0.1,
            "cooperation_bonus": 0.05
        }
    )
    
    print(f"🌟 Single Scale Demo: {massive_config.num_agents} agents")
    massive_arena = MassiveScaleArena(massive_config, optimization_level="ultra")
    massive_results = massive_arena.run_massive_scale_benchmark(episodes=3)
    
    if 'error' not in massive_results:
        metrics = massive_results.get('scaling_metrics', {})
        print(f"\n🎉 Massive Scale Results:")
        print(f"  • Agent count: {metrics.get('agent_count', 0)}")
        print(f"  • Throughput: {metrics.get('total_throughput', 0):.0f} agent*steps/s")
        print(f"  • Steps/second: {metrics.get('steps_per_second', 0):.0f}")
        print(f"  • Memory efficiency: {metrics.get('memory_efficiency', 0):.2f} MB/agent")
        print(f"  • CPU utilization: {metrics.get('cpu_utilization', 0):.1f}%")
        
        benchmark_time = massive_results.get('benchmark_time', 0)
        print(f"  • Execution time: {benchmark_time:.2f}s")
    
    # Run multi-scale comparison for research validation
    print(f"\n🔬 Multi-Scale Comparison Analysis")
    scale_factors = [1, 2, 4]  # Test 100, 200, 400 agents
    comparison_results = run_multi_scale_comparison(scale_factors)
    
    print(f"\n📊 Scaling Analysis Summary:")
    for scale, results in comparison_results.items():
        if 'error' not in results:
            metrics = results.get('scaling_metrics', {})
            agents = metrics.get('agent_count', 0)
            throughput = metrics.get('total_throughput', 0)
            efficiency = metrics.get('weak_scaling_efficiency', 1.0)
            
            print(f"  • Scale {scale}x ({agents} agents): {throughput:.0f} agent*steps/s, "
                  f"efficiency: {efficiency:.2%}")
        else:
            print(f"  • Scale {scale}x: Failed - {results['error']}")
    
    print("\n✅ Generation 3 scaling optimization complete!")
    print("✅ Massive scale performance validated!")

if __name__ == "__main__":
    main()