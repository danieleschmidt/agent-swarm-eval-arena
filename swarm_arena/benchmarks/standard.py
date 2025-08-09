"""Standard benchmarking suite for agent evaluation."""

import time
import numpy as np
from typing import List, Dict, Any, Type, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.arena import Arena, SimulationResults
from ..core.config import SwarmConfig
from ..core.agent import BaseAgent, Agent, CooperativeAgent, CompetitiveAgent
from ..core.environment import ForagingEnvironment
from ..exceptions import SimulationError
from ..utils.logging import get_logger
from ..utils.seeding import set_global_seed

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    
    agent_name: str
    environment_name: str
    seed: int
    mean_reward: float
    std_reward: float
    efficiency: float
    fairness_index: float
    survival_rate: float
    cooperation_score: float
    total_steps: int
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "environment_name": self.environment_name,
            "seed": self.seed,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "efficiency": self.efficiency,
            "fairness_index": self.fairness_index,
            "survival_rate": self.survival_rate,
            "cooperation_score": self.cooperation_score,
            "total_steps": self.total_steps,
            "execution_time": self.execution_time
        }


class StandardBenchmark:
    """Standard benchmarking suite for multi-agent evaluation."""
    
    def __init__(self, 
                 base_config: Optional[SwarmConfig] = None,
                 num_episodes: int = 10,
                 num_seeds: int = 5,
                 parallel_execution: bool = True,
                 max_workers: int = 4) -> None:
        """Initialize benchmark suite.
        
        Args:
            base_config: Base configuration for benchmarks
            num_episodes: Episodes per benchmark run
            num_seeds: Number of random seeds to test
            parallel_execution: Whether to run benchmarks in parallel
            max_workers: Maximum parallel workers
        """
        self.base_config = base_config or SwarmConfig(
            num_agents=50,
            episode_length=500,
            arena_size=(800, 600)
        )
        self.num_episodes = num_episodes
        self.num_seeds = num_seeds
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        
        logger.info(f"Standard benchmark initialized (episodes={num_episodes}, seeds={num_seeds})")
    
    def run_single_benchmark(self, 
                           agent_class: Type[BaseAgent],
                           environment_class: Type,
                           seed: int,
                           config_overrides: Optional[Dict[str, Any]] = None) -> BenchmarkResult:
        """Run a single benchmark configuration.
        
        Args:
            agent_class: Agent class to benchmark
            environment_class: Environment class to use
            seed: Random seed
            config_overrides: Configuration overrides
            
        Returns:
            Benchmark result
        """
        try:
            # Set global seed
            set_global_seed(seed)
            
            # Create configuration
            config = self.base_config.copy(seed=seed)
            if config_overrides:
                for key, value in config_overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Create environment and arena
            environment = environment_class(config)
            arena = Arena(config, environment)
            
            # Add agents
            arena.add_agents(agent_class, count=config.num_agents)
            
            # Run benchmark
            start_time = time.time()
            results = arena.run(episodes=self.num_episodes, verbose=False)
            execution_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_metrics(results, arena)
            
            # Create benchmark result
            benchmark_result = BenchmarkResult(
                agent_name=agent_class.__name__,
                environment_name=environment_class.__name__,
                seed=seed,
                execution_time=execution_time,
                **metrics
            )
            
            logger.debug(f"Completed benchmark: {agent_class.__name__} in {environment_class.__name__} (seed={seed})")
            return benchmark_result
            
        except Exception as e:
            logger.error(f"Benchmark failed: {agent_class.__name__} in {environment_class.__name__} (seed={seed}): {str(e)}")
            # Return failed result
            return BenchmarkResult(
                agent_name=agent_class.__name__,
                environment_name=environment_class.__name__,
                seed=seed,
                mean_reward=0.0,
                std_reward=0.0,
                efficiency=0.0,
                fairness_index=0.0,
                survival_rate=0.0,
                cooperation_score=0.0,
                total_steps=0,
                execution_time=0.0
            )
    
    def run_all(self, 
               agent_classes: List[Type[BaseAgent]],
               environments: Optional[List[str]] = None,
               metrics: Optional[List[str]] = None) -> List[BenchmarkResult]:
        """Run comprehensive benchmark suite.
        
        Args:
            agent_classes: List of agent classes to benchmark
            environments: List of environment names
            metrics: List of metrics to compute
            
        Returns:
            List of benchmark results
        """
        try:
            if environments is None:
                environments = ["foraging_standard", "foraging_sparse"]
            
            if metrics is None:
                metrics = ["reward", "efficiency", "fairness", "survival", "cooperation"]
            
            # Define environment configurations
            env_configs = {
                "foraging_standard": (ForagingEnvironment, {}),
                "foraging_sparse": (ForagingEnvironment, {"resource_spawn_rate": 0.02}),
                "foraging_dense": (ForagingEnvironment, {"resource_spawn_rate": 0.2}),
                "pursuit_evasion": (ForagingEnvironment, {})  # Simplified for now
            }
            
            # Generate all benchmark configurations
            benchmark_configs = []
            for agent_class in agent_classes:
                for env_name in environments:
                    if env_name not in env_configs:
                        logger.warning(f"Unknown environment: {env_name}, using ForagingEnvironment")
                        env_class, config_overrides = ForagingEnvironment, {}
                    else:
                        env_class, config_overrides = env_configs[env_name]
                    
                    for seed in range(self.num_seeds):
                        benchmark_configs.append((
                            agent_class,
                            env_class,
                            seed,
                            config_overrides
                        ))
            
            logger.info(f"Running {len(benchmark_configs)} benchmark configurations")
            
            # Execute benchmarks
            if self.parallel_execution and len(benchmark_configs) > 1:
                results = self._run_parallel_benchmarks(benchmark_configs)
            else:
                results = self._run_sequential_benchmarks(benchmark_configs)
            
            self.results.extend(results)
            
            logger.info(f"Completed {len(results)} benchmarks")
            return results
            
        except Exception as e:
            raise SimulationError(f"Benchmark suite execution failed: {str(e)}")
    
    def _run_parallel_benchmarks(self, configs: List[tuple]) -> List[BenchmarkResult]:
        """Run benchmarks in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all benchmark jobs
            future_to_config = {
                executor.submit(
                    self.run_single_benchmark,
                    agent_class, env_class, seed, config_overrides
                ): (agent_class.__name__, env_class.__name__, seed)
                for agent_class, env_class, seed, config_overrides in configs
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_config):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if completed % 5 == 0:
                        logger.info(f"Completed {completed}/{len(configs)} benchmarks")
                        
                except Exception as e:
                    agent_name, env_name, seed = future_to_config[future]
                    logger.error(f"Parallel benchmark failed: {agent_name} in {env_name} (seed={seed}): {str(e)}")
        
        return results
    
    def _run_sequential_benchmarks(self, configs: List[tuple]) -> List[BenchmarkResult]:
        """Run benchmarks sequentially."""
        results = []
        
        for i, (agent_class, env_class, seed, config_overrides) in enumerate(configs):
            result = self.run_single_benchmark(agent_class, env_class, seed, config_overrides)
            results.append(result)
            
            if (i + 1) % 5 == 0:
                logger.info(f"Completed {i + 1}/{len(configs)} benchmarks")
        
        return results
    
    def _calculate_metrics(self, results: SimulationResults, arena: Arena) -> Dict[str, float]:
        """Calculate benchmark metrics from simulation results."""
        try:
            # Basic reward metrics
            all_rewards = []
            for agent_rewards in results.episode_rewards.values():
                all_rewards.extend(agent_rewards)
            
            mean_reward = np.mean(all_rewards) if all_rewards else 0.0
            std_reward = np.std(all_rewards) if all_rewards else 0.0
            
            # Efficiency metric (resources collected per agent)
            total_resources = sum(
                stats.get("resources_collected", 0)
                for stats in results.agent_stats.values()
            )
            efficiency = total_resources / len(arena.agents) if arena.agents else 0.0
            
            # Fairness index (from results or calculate Gini coefficient)
            fairness_index = results.fairness_index if results.fairness_index is not None else 0.0
            
            # Survival rate
            alive_agents = sum(
                1 for stats in results.agent_stats.values() 
                if stats.get("alive", True)
            )
            survival_rate = alive_agents / len(arena.agents) if arena.agents else 0.0
            
            # Cooperation score (based on agent clustering/coordination)
            cooperation_score = self._calculate_cooperation_score(results, arena)
            
            return {
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "efficiency": float(efficiency),
                "fairness_index": float(fairness_index),
                "survival_rate": float(survival_rate),
                "cooperation_score": float(cooperation_score),
                "total_steps": results.total_steps
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "efficiency": 0.0,
                "fairness_index": 0.0,
                "survival_rate": 0.0,
                "cooperation_score": 0.0,
                "total_steps": 0
            }
    
    def _calculate_cooperation_score(self, results: SimulationResults, arena: Arena) -> float:
        """Calculate cooperation score based on agent behavior."""
        try:
            # Simple cooperation metric based on reward variance
            # Lower variance indicates more cooperative behavior
            agent_totals = [sum(rewards) for rewards in results.episode_rewards.values()]
            
            if len(agent_totals) < 2:
                return 0.0
            
            mean_total = np.mean(agent_totals)
            if mean_total == 0:
                return 0.0
            
            # Normalized inverse of coefficient of variation
            cv = np.std(agent_totals) / mean_total
            cooperation_score = max(0.0, 1.0 - cv)
            
            return float(cooperation_score)
            
        except Exception as e:
            logger.error(f"Error calculating cooperation score: {str(e)}")
            return 0.0
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all benchmark results."""
        if not self.results:
            return {"status": "no_results", "message": "No benchmark results available"}
        
        try:
            # Group results by agent and environment
            grouped = {}
            for result in self.results:
                key = (result.agent_name, result.environment_name)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(result)
            
            # Calculate statistics for each group
            summary = {}
            for (agent_name, env_name), group_results in grouped.items():
                key = f"{agent_name}_{env_name}"
                
                # Extract metrics
                rewards = [r.mean_reward for r in group_results]
                efficiencies = [r.efficiency for r in group_results]
                fairness_scores = [r.fairness_index for r in group_results]
                
                summary[key] = {
                    "agent": agent_name,
                    "environment": env_name,
                    "num_runs": len(group_results),
                    "mean_reward": {
                        "mean": np.mean(rewards),
                        "std": np.std(rewards),
                        "min": np.min(rewards),
                        "max": np.max(rewards)
                    },
                    "efficiency": {
                        "mean": np.mean(efficiencies),
                        "std": np.std(efficiencies)
                    },
                    "fairness": {
                        "mean": np.mean(fairness_scores),
                        "std": np.std(fairness_scores)
                    },
                    "avg_execution_time": np.mean([r.execution_time for r in group_results])
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def to_latex_table(self, 
                      results: Optional[List[BenchmarkResult]] = None,
                      caption: str = "Benchmark Results",
                      label: str = "tab:benchmark_results") -> str:
        """Generate LaTeX table from benchmark results."""
        if results is None:
            results = self.results
        
        if not results:
            return "% No benchmark results available"
        
        try:
            # Group and average results
            grouped = {}
            for result in results:
                key = (result.agent_name, result.environment_name)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(result)
            
            # Generate LaTeX table
            latex_lines = [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\begin{tabular}{|l|l|c|c|c|c|}",
                "\\hline",
                "Agent & Environment & Mean Reward & Efficiency & Fairness & Survival \\\\",
                "\\hline"
            ]
            
            for (agent_name, env_name), group_results in sorted(grouped.items()):
                mean_reward = np.mean([r.mean_reward for r in group_results])
                efficiency = np.mean([r.efficiency for r in group_results])
                fairness = np.mean([r.fairness_index for r in group_results])
                survival = np.mean([r.survival_rate for r in group_results])
                
                line = f"{agent_name} & {env_name} & {mean_reward:.3f} & {efficiency:.3f} & {fairness:.3f} & {survival:.3f} \\\\"
                latex_lines.append(line)
            
            latex_lines.extend([
                "\\hline",
                "\\end{tabular}",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                "\\end{table}"
            ])
            
            return "\n".join(latex_lines)
            
        except Exception as e:
            logger.error(f"Error generating LaTeX table: {str(e)}")
            return f"% Error generating table: {str(e)}"
    
    def export_results(self, filename: str, format: str = "json") -> None:
        """Export benchmark results to file."""
        try:
            import json
            
            if format.lower() == "json":
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
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported {len(self.results)} benchmark results to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            raise