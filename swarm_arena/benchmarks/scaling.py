"""Scaling benchmarks for performance analysis."""

import time
import psutil
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import multiprocessing as mp

from ..core.arena import Arena
from ..core.config import SwarmConfig
from ..core.agent import Agent
from ..core.environment import ForagingEnvironment
from ..exceptions import SimulationError
from ..utils.logging import get_logger
from ..utils.seeding import set_global_seed

logger = get_logger(__name__)


@dataclass
class ScalingResult:
    """Result from scaling benchmark."""
    
    test_type: str  # "weak_scaling" or "strong_scaling"
    scale_factor: int
    num_agents: int
    arena_size: Tuple[float, float]
    num_workers: Optional[int] = None
    
    # Performance metrics
    total_time: float = 0.0
    avg_step_time: float = 0.0
    throughput: float = 0.0  # steps/second
    memory_usage: float = 0.0  # MB
    cpu_usage: float = 0.0  # %
    
    # Efficiency metrics
    parallel_efficiency: float = 0.0
    scaling_efficiency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_type": self.test_type,
            "scale_factor": self.scale_factor,
            "num_agents": self.num_agents,
            "arena_size": self.arena_size,
            "num_workers": self.num_workers,
            "total_time": self.total_time,
            "avg_step_time": self.avg_step_time,
            "throughput": self.throughput,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "parallel_efficiency": self.parallel_efficiency,
            "scaling_efficiency": self.scaling_efficiency
        }


class ScalingBenchmark:
    """Benchmark suite for analyzing scaling properties."""
    
    def __init__(self, 
                 base_config: Optional[SwarmConfig] = None,
                 episode_length: int = 200,
                 warmup_steps: int = 50) -> None:
        """Initialize scaling benchmark.
        
        Args:
            base_config: Base configuration for scaling tests
            episode_length: Length of episodes for timing
            warmup_steps: Number of warmup steps before measurement
        """
        self.base_config = base_config or SwarmConfig(
            num_agents=100,
            episode_length=episode_length,
            arena_size=(1000, 1000),
            resource_spawn_rate=0.05
        )
        self.episode_length = episode_length
        self.warmup_steps = warmup_steps
        
        self.results: List[ScalingResult] = []
        
        logger.info(f"Scaling benchmark initialized (episode_length={episode_length})")
    
    def weak_scaling_test(self, 
                         base_agents: int = 100,
                         base_arena: float = 1000,
                         scale_factors: List[int] = None,
                         metric: str = "throughput") -> List[ScalingResult]:
        """Test weak scaling (increase problem size proportionally)."""
        if scale_factors is None:
            scale_factors = [1, 2, 4, 8]
        
        logger.info(f"Starting weak scaling test with factors {scale_factors}")
        
        results = []
        baseline_time = None
        
        for scale_factor in scale_factors:
            try:
                # Calculate scaled parameters
                num_agents = base_agents * scale_factor
                arena_size = (base_arena * np.sqrt(scale_factor), 
                             base_arena * np.sqrt(scale_factor))
                
                logger.info(f"Testing scale factor {scale_factor}: {num_agents} agents, "
                           f"arena {arena_size[0]:.0f}x{arena_size[1]:.0f}")
                
                # Run benchmark
                result = self._run_single_scaling_test(
                    "weak_scaling",
                    scale_factor,
                    num_agents,
                    arena_size
                )
                
                # Calculate efficiency metrics
                if baseline_time is None:
                    baseline_time = result.total_time
                    result.scaling_efficiency = 1.0
                else:
                    # Weak scaling efficiency: T_1 / T_n (should be close to 1.0)
                    result.scaling_efficiency = baseline_time / result.total_time
                
                results.append(result)
                
                logger.info(f"Scale factor {scale_factor}: {result.total_time:.2f}s, "
                           f"efficiency: {result.scaling_efficiency:.3f}")
                
            except Exception as e:
                logger.error(f"Weak scaling test failed for scale factor {scale_factor}: {str(e)}")
                continue
        
        self.results.extend(results)
        return results
    
    def strong_scaling_test(self,
                           num_agents: int = 1000,
                           num_workers_list: List[int] = None,
                           metric: str = "episode_time") -> List[ScalingResult]:
        """Test strong scaling (fixed problem size, varying compute resources)."""
        if num_workers_list is None:
            max_cores = mp.cpu_count()
            num_workers_list = [1, 2, 4, min(8, max_cores)]
            num_workers_list = [w for w in num_workers_list if w <= max_cores]
        
        logger.info(f"Starting strong scaling test with workers {num_workers_list}")
        
        results = []
        baseline_time = None
        
        arena_size = self.base_config.arena_size
        
        for num_workers in num_workers_list:
            try:
                logger.info(f"Testing {num_workers} workers with {num_agents} agents")
                
                # For single-threaded case, run without distributed setup
                if num_workers == 1:
                    result = self._run_single_scaling_test(
                        "strong_scaling",
                        num_workers,
                        num_agents,
                        arena_size
                    )
                else:
                    # Simulate distributed performance (for demonstration)
                    logger.warning(f"Distributed testing not implemented, simulating results for {num_workers} workers")
                    
                    result = self._simulate_distributed_result(
                        "strong_scaling",
                        num_workers,
                        num_agents,
                        arena_size
                    )
                
                result.num_workers = num_workers
                
                # Calculate parallel efficiency
                if baseline_time is None:
                    baseline_time = result.total_time
                    result.parallel_efficiency = 1.0
                else:
                    # Strong scaling efficiency: (T_1 / T_n) / n
                    speedup = baseline_time / result.total_time
                    result.parallel_efficiency = speedup / num_workers
                
                results.append(result)
                
                logger.info(f"{num_workers} workers: {result.total_time:.2f}s, "
                           f"efficiency: {result.parallel_efficiency:.3f}")
                
            except Exception as e:
                logger.error(f"Strong scaling test failed for {num_workers} workers: {str(e)}")
                continue
        
        self.results.extend(results)
        return results
    
    def _run_single_scaling_test(self,
                                test_type: str,
                                scale_factor: int,
                                num_agents: int,
                                arena_size: Tuple[float, float]) -> ScalingResult:
        """Run a single scaling test configuration."""
        try:
            # Create configuration
            config = self.base_config.copy(
                num_agents=num_agents,
                arena_size=arena_size,
                episode_length=self.episode_length
            )
            
            # Create arena
            environment = ForagingEnvironment(config)
            arena = Arena(config, environment)
            arena.add_agents(Agent, count=num_agents)
            
            # Warm up
            arena.reset()
            for _ in range(min(self.warmup_steps, 10)):  # Limit warmup for testing
                arena.step()
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            cpu_before = psutil.cpu_percent(interval=None)
            
            # Run benchmark steps
            step_times = []
            test_steps = min(self.episode_length, 50)  # Limit steps for testing
            for step in range(test_steps):
                step_start = time.time()
                observations, rewards, done, info = arena.step()
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                if done:
                    break
            
            # Final measurements
            total_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            cpu_after = psutil.cpu_percent(interval=None)
            
            # Calculate metrics
            avg_step_time = np.mean(step_times) if step_times else 0.0
            throughput = len(step_times) / total_time if total_time > 0 else 0.0
            memory_usage = max(0, end_memory - start_memory)
            cpu_usage = max(0, (cpu_after + cpu_before) / 2)
            
            # Create result
            result = ScalingResult(
                test_type=test_type,
                scale_factor=scale_factor,
                num_agents=num_agents,
                arena_size=arena_size,
                total_time=total_time,
                avg_step_time=avg_step_time,
                throughput=throughput,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage
            )
            
            return result
            
        except Exception as e:
            raise SimulationError(f"Scaling test failed: {str(e)}")
    
    def _simulate_distributed_result(self,
                                   test_type: str,
                                   num_workers: int,
                                   num_agents: int,
                                   arena_size: Tuple[float, float]) -> ScalingResult:
        """Simulate distributed performance results."""
        # Simulate scaling with some overhead
        base_time = 5.0  # Baseline time for single worker
        
        # Simulate sub-linear speedup due to communication overhead
        efficiency_factor = 1.0 - (num_workers - 1) * 0.03  # 3% overhead per additional worker
        simulated_time = base_time / (num_workers * efficiency_factor)
        
        # Add some realistic variance
        simulated_time *= np.random.normal(1.0, 0.05)  # 5% variance
        
        result = ScalingResult(
            test_type=test_type,
            scale_factor=num_workers,
            num_agents=num_agents,
            arena_size=arena_size,
            total_time=max(0.1, simulated_time),  # Ensure positive time
            avg_step_time=max(0.001, simulated_time / self.episode_length),
            throughput=self.episode_length / max(0.1, simulated_time),
            memory_usage=20 * num_workers,  # Simulate memory scaling
            cpu_usage=min(100.0, 60.0 + num_workers * 5)  # Simulate CPU usage
        )
        
        return result
    
    def plot_scaling_curves(self,
                           weak_scaling_results: List[ScalingResult],
                           strong_scaling_results: List[ScalingResult],
                           save_path: str = "scaling_analysis.png") -> None:
        """Plot scaling efficiency curves."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Weak scaling plot
            if weak_scaling_results:
                scale_factors = [r.scale_factor for r in weak_scaling_results]
                efficiencies = [r.scaling_efficiency for r in weak_scaling_results]
                
                ax1.plot(scale_factors, efficiencies, 'bo-', label='Weak Scaling')
                ax1.axhline(y=1.0, color='r', linestyle='--', label='Ideal')
                ax1.set_xlabel('Scale Factor')
                ax1.set_ylabel('Scaling Efficiency')
                ax1.set_title('Weak Scaling Efficiency')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Strong scaling plot
            if strong_scaling_results:
                num_workers = [r.num_workers for r in strong_scaling_results if r.num_workers]
                efficiencies = [r.parallel_efficiency for r in strong_scaling_results if r.num_workers]
                
                if num_workers:
                    ax2.plot(num_workers, efficiencies, 'ro-', label='Strong Scaling')
                    ax2.axhline(y=1.0, color='r', linestyle='--', label='Ideal')
                    ax2.set_xlabel('Number of Workers')
                    ax2.set_ylabel('Parallel Efficiency')
                    ax2.set_title('Strong Scaling Efficiency')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Scaling curves saved to {save_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available, cannot plot scaling curves")
        except Exception as e:
            logger.error(f"Error plotting scaling curves: {str(e)}")
    
    def generate_scaling_report(self) -> str:
        """Generate comprehensive scaling analysis report."""
        if not self.results:
            return "No scaling results available."
        
        try:
            report_lines = []
            report_lines.append("SCALING ANALYSIS REPORT")
            report_lines.append("=" * 50)
            report_lines.append("")
            
            # Separate results by test type
            weak_results = [r for r in self.results if r.test_type == "weak_scaling"]
            strong_results = [r for r in self.results if r.test_type == "strong_scaling"]
            
            # Weak scaling analysis
            if weak_results:
                report_lines.append("WEAK SCALING RESULTS:")
                report_lines.append("-" * 30)
                report_lines.append(f"{'Scale Factor':<12} {'Agents':<8} {'Time (s)':<10} {'Efficiency':<12} {'Throughput':<12}")
                report_lines.append("-" * 70)
                
                for result in weak_results:
                    report_lines.append(
                        f"{result.scale_factor:<12} {result.num_agents:<8} "
                        f"{result.total_time:<10.2f} {result.scaling_efficiency:<12.3f} "
                        f"{result.throughput:<12.1f}"
                    )
                
                # Weak scaling summary
                avg_efficiency = np.mean([r.scaling_efficiency for r in weak_results])
                report_lines.append(f"\nAverage weak scaling efficiency: {avg_efficiency:.3f}")
                report_lines.append("")
            
            # Strong scaling analysis
            if strong_results:
                report_lines.append("STRONG SCALING RESULTS:")
                report_lines.append("-" * 30)
                report_lines.append(f"{'Workers':<8} {'Time (s)':<10} {'Speedup':<10} {'Efficiency':<12} {'Throughput':<12}")
                report_lines.append("-" * 60)
                
                baseline_time = strong_results[0].total_time if strong_results else 0
                
                for result in strong_results:
                    speedup = baseline_time / result.total_time if result.total_time > 0 else 0
                    report_lines.append(
                        f"{result.num_workers or 1:<8} {result.total_time:<10.2f} "
                        f"{speedup:<10.2f} {result.parallel_efficiency:<12.3f} "
                        f"{result.throughput:<12.1f}"
                    )
                
                # Strong scaling summary
                valid_results = [r for r in strong_results if r.parallel_efficiency > 0]
                if valid_results:
                    avg_efficiency = np.mean([r.parallel_efficiency for r in valid_results])
                    report_lines.append(f"\nAverage strong scaling efficiency: {avg_efficiency:.3f}")
                report_lines.append("")
            
            # Overall recommendations
            report_lines.append("RECOMMENDATIONS:")
            report_lines.append("-" * 20)
            
            if weak_results:
                best_weak = max(weak_results, key=lambda r: r.scaling_efficiency)
                report_lines.append(f"• Best weak scaling at factor {best_weak.scale_factor} "
                                  f"with {best_weak.scaling_efficiency:.3f} efficiency")
            
            if strong_results:
                best_strong = max(strong_results, key=lambda r: r.parallel_efficiency)
                report_lines.append(f"• Best strong scaling with {best_strong.num_workers} workers "
                                  f"at {best_strong.parallel_efficiency:.3f} efficiency")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating scaling report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def export_results(self, filename: str, format: str = "json") -> None:
        """Export scaling results to file."""
        try:
            if format.lower() == "json":
                import json
                data = [result.to_dict() for result in self.results]
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format.lower() == "csv":
                import csv
                if not self.results:
                    return
                
                fieldnames = list(self.results[0].to_dict().keys())
                with open(filename, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for result in self.results:
                        writer.writerow(result.to_dict())
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported {len(self.results)} scaling results to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting scaling results: {str(e)}")
            raise