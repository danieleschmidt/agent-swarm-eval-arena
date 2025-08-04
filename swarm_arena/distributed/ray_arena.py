"""Ray-based distributed arena for large-scale simulations."""

import ray
import numpy as np
from typing import Dict, List, Any, Optional, Type, Tuple
import time
from dataclasses import dataclass

from ..core.arena import Arena, SimulationResults
from ..core.config import SwarmConfig
from ..core.agent import BaseAgent
from ..core.environment import Environment
from ..exceptions import RayError, SimulationError
from ..utils.logging import get_logger
from .worker import ArenaWorker

logger = get_logger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed execution."""
    
    num_workers: int = 4
    agents_per_worker: int = 250
    gpu_per_worker: float = 0.0
    cpu_per_worker: int = 2
    memory_per_worker: int = 2048  # MB
    use_placement_groups: bool = True
    synchronization_method: str = "barrier"  # "barrier" or "async"


class DistributedArena:
    """Distributed arena using Ray for massive scale simulations."""
    
    def __init__(self, 
                 config: SwarmConfig,
                 distributed_config: Optional[DistributedConfig] = None,
                 environment: Optional[Environment] = None) -> None:
        """Initialize distributed arena.
        
        Args:
            config: Swarm configuration
            distributed_config: Distributed execution configuration
            environment: Environment instance
            
        Raises:
            RayError: If Ray initialization fails
        """
        self.config = config
        self.distributed_config = distributed_config or DistributedConfig()
        self.environment = environment
        
        # Initialize Ray if not already running
        if not ray.is_initialized():
            try:
                ray.init(ignore_reinit_error=True)
                logger.info("Initialized Ray cluster")
            except Exception as e:
                raise RayError(f"Failed to initialize Ray: {str(e)}")
        
        # Calculate worker distribution
        self._calculate_worker_distribution()
        
        # Create worker pool
        self.workers: List[ray.ObjectRef] = []
        self.placement_group: Optional[ray.PlacementGroup] = None
        
        self._create_workers()
        
        logger.info(f"Distributed arena initialized with {len(self.workers)} workers")
    
    def _calculate_worker_distribution(self) -> None:
        """Calculate optimal worker and agent distribution."""
        total_agents = self.config.num_agents
        
        # Adjust number of workers based on total agents
        if total_agents < self.distributed_config.agents_per_worker:
            self.distributed_config.num_workers = 1
            self.distributed_config.agents_per_worker = total_agents
        else:
            # Calculate optimal distribution
            optimal_workers = max(1, total_agents // self.distributed_config.agents_per_worker)
            self.distributed_config.num_workers = min(
                self.distributed_config.num_workers, 
                optimal_workers
            )
            
            # Recalculate agents per worker
            self.distributed_config.agents_per_worker = total_agents // self.distributed_config.num_workers
        
        logger.info(f"Worker distribution: {self.distributed_config.num_workers} workers, "
                   f"{self.distributed_config.agents_per_worker} agents per worker")
    
    def _create_workers(self) -> None:
        """Create Ray worker pool."""
        try:\n            # Create placement group for resource management\n            if self.distributed_config.use_placement_groups:\n                bundles = []\n                for i in range(self.distributed_config.num_workers):\n                    bundle = {\n                        \"CPU\": self.distributed_config.cpu_per_worker,\n                        \"memory\": self.distributed_config.memory_per_worker * 1024 * 1024  # Convert to bytes\n                    }\n                    \n                    if self.distributed_config.gpu_per_worker > 0:\n                        bundle[\"GPU\"] = self.distributed_config.gpu_per_worker\n                    \n                    bundles.append(bundle)\n                \n                self.placement_group = ray.util.placement_group(\n                    bundles, strategy=\"SPREAD\"\n                )\n                ray.get(self.placement_group.ready())\n                logger.info(\"Created placement group for workers\")\n            \n            # Create workers\n            for worker_id in range(self.distributed_config.num_workers):\n                # Calculate agent range for this worker\n                start_id = worker_id * self.distributed_config.agents_per_worker\n                end_id = min(\n                    (worker_id + 1) * self.distributed_config.agents_per_worker,\n                    self.config.num_agents\n                )\n                \n                # Create worker-specific config\n                worker_config = self.config.copy(\n                    num_agents=end_id - start_id\n                )\n                \n                # Resource specification\n                resources = {\n                    \"num_cpus\": self.distributed_config.cpu_per_worker,\n                    \"memory\": self.distributed_config.memory_per_worker * 1024 * 1024\n                }\n                \n                if self.distributed_config.gpu_per_worker > 0:\n                    resources[\"num_gpus\"] = self.distributed_config.gpu_per_worker\n                \n                # Create worker\n                if self.placement_group:\n                    worker = ArenaWorker.options(\n                        **resources,\n                        placement_group=self.placement_group,\n                        placement_group_bundle_index=worker_id\n                    ).remote(\n                        worker_id=worker_id,\n                        config=worker_config,\n                        agent_id_offset=start_id,\n                        environment_class=type(self.environment) if self.environment else None\n                    )\n                else:\n                    worker = ArenaWorker.options(**resources).remote(\n                        worker_id=worker_id,\n                        config=worker_config,\n                        agent_id_offset=start_id,\n                        environment_class=type(self.environment) if self.environment else None\n                    )\n                \n                self.workers.append(worker)\n                \n        except Exception as e:\n            raise RayError(f\"Failed to create workers: {str(e)}\")\n    \n    def add_agents(self, agent_class: Type[BaseAgent], count: int, **agent_kwargs: Any) -> None:\n        \"\"\"Add agents distributed across workers.\n        \n        Args:\n            agent_class: Agent class to instantiate\n            count: Total number of agents to add\n            **agent_kwargs: Additional agent parameters\n        \"\"\"\n        try:\n            # Distribute agents across workers\n            agents_per_worker = count // len(self.workers)\n            remaining_agents = count % len(self.workers)\n            \n            futures = []\n            for i, worker in enumerate(self.workers):\n                worker_agent_count = agents_per_worker\n                if i < remaining_agents:\n                    worker_agent_count += 1\n                \n                if worker_agent_count > 0:\n                    future = worker.add_agents.remote(\n                        agent_class, worker_agent_count, **agent_kwargs\n                    )\n                    futures.append(future)\n            \n            # Wait for all workers to complete\n            ray.get(futures)\n            logger.info(f\"Added {count} agents of type {agent_class.__name__} across workers\")\n            \n        except Exception as e:\n            raise SimulationError(f\"Failed to add agents to distributed arena: {str(e)}\")\n    \n    def reset(self) -> Dict[str, Any]:\n        \"\"\"Reset all workers.\n        \n        Returns:\n            Combined reset information\n        \"\"\"\n        try:\n            # Reset all workers in parallel\n            reset_futures = [worker.reset.remote() for worker in self.workers]\n            reset_results = ray.get(reset_futures)\n            \n            # Combine results\n            total_agents = sum(result[\"num_agents\"] for result in reset_results)\n            \n            combined_result = {\n                \"num_agents\": total_agents,\n                \"num_workers\": len(self.workers),\n                \"arena_size\": self.config.arena_size,\n                \"worker_results\": reset_results\n            }\n            \n            logger.info(f\"Reset distributed arena with {total_agents} agents across {len(self.workers)} workers\")\n            return combined_result\n            \n        except Exception as e:\n            raise SimulationError(f\"Failed to reset distributed arena: {str(e)}\")\n    \n    def step(self) -> Tuple[Dict[str, Any], Dict[str, Any], bool, Dict[str, Any]]:\n        \"\"\"Execute one distributed simulation step.\n        \n        Returns:\n            Combined step results from all workers\n        \"\"\"\n        try:\n            # Execute step on all workers\n            step_futures = [worker.step.remote() for worker in self.workers]\n            \n            if self.distributed_config.synchronization_method == \"barrier\":\n                # Synchronous execution - wait for all workers\n                step_results = ray.get(step_futures)\n            else:\n                # Asynchronous execution - collect results as they complete\n                step_results = []\n                while step_futures:\n                    ready, step_futures = ray.wait(step_futures, num_returns=1)\n                    step_results.extend(ray.get(ready))\n            \n            # Combine results from all workers\n            combined_observations = {}\n            combined_rewards = {}\n            all_done = True\n            combined_info = {\n                \"worker_count\": len(step_results),\n                \"total_agents\": 0,\n                \"total_active_agents\": 0,\n                \"worker_stats\": []\n            }\n            \n            for worker_id, (obs, rewards, done, info) in enumerate(step_results):\n                # Merge observations and rewards\n                combined_observations.update(obs)\n                combined_rewards.update(rewards)\n                \n                # All workers must be done for overall done\n                all_done = all_done and done\n                \n                # Aggregate info\n                combined_info[\"total_agents\"] += info.get(\"agent_count\", 0)\n                combined_info[\"total_active_agents\"] += info.get(\"active_agents\", 0)\n                combined_info[\"worker_stats\"].append({\n                    \"worker_id\": worker_id,\n                    \"step_time\": info.get(\"step_time\", 0),\n                    \"active_agents\": info.get(\"active_agents\", 0)\n                })\n            \n            # Calculate average step time\n            step_times = [stats[\"step_time\"] for stats in combined_info[\"worker_stats\"]]\n            combined_info[\"avg_step_time\"] = np.mean(step_times) if step_times else 0\n            combined_info[\"max_step_time\"] = max(step_times) if step_times else 0\n            \n            return combined_observations, combined_rewards, all_done, combined_info\n            \n        except Exception as e:\n            raise SimulationError(f\"Distributed step execution failed: {str(e)}\")\n    \n    def run(self, episodes: int = 1, verbose: bool = True) -> SimulationResults:\n        \"\"\"Run distributed simulation.\n        \n        Args:\n            episodes: Number of episodes to run\n            verbose: Whether to print progress\n            \n        Returns:\n            Combined simulation results\n        \"\"\"\n        try:\n            all_episode_rewards = {}\n            total_steps = 0\n            \n            for episode in range(episodes):\n                if verbose:\n                    print(f\"Running distributed episode {episode + 1}/{episodes}\")\n                \n                # Reset for new episode\n                self.reset()\n                \n                # Run episode\n                episode_start = time.time()\n                done = False\n                step_count = 0\n                \n                while not done:\n                    step_start = time.time()\n                    observations, rewards, done, info = self.step()\n                    step_time = time.time() - step_start\n                    \n                    step_count += 1\n                    \n                    if verbose and step_count % 100 == 0:\n                        active_agents = info.get(\"total_active_agents\", 0)\n                        avg_step_time = info.get(\"avg_step_time\", 0)\n                        print(f\"  Step {step_count}: {active_agents} active agents, \"\n                              f\"avg step time: {avg_step_time:.3f}s\")\n                \n                episode_time = time.time() - episode_start\n                total_steps += step_count\n                \n                if verbose:\n                    print(f\"  Episode completed in {episode_time:.2f}s ({step_count} steps)\")\n                \n                # Collect episode rewards from workers\n                reward_futures = [worker.get_episode_rewards.remote() for worker in self.workers]\n                worker_rewards = ray.get(reward_futures)\n                \n                # Merge episode rewards\n                for worker_reward_dict in worker_rewards:\n                    for agent_id, episode_total in worker_reward_dict.items():\n                        if agent_id not in all_episode_rewards:\n                            all_episode_rewards[agent_id] = []\n                        all_episode_rewards[agent_id].append(episode_total)\n            \n            # Get final agent stats from workers\n            stats_futures = [worker.get_agent_stats.remote() for worker in self.workers]\n            worker_stats = ray.get(stats_futures)\n            \n            # Combine agent stats\n            combined_agent_stats = {}\n            for worker_stat_dict in worker_stats:\n                combined_agent_stats.update(worker_stat_dict)\n            \n            # Create combined results\n            results = SimulationResults(\n                episode_rewards=all_episode_rewards,\n                agent_stats=combined_agent_stats,\n                environment_stats={\"distributed_workers\": len(self.workers)},\n                total_steps=total_steps,\n                episode_length=total_steps // episodes if episodes > 0 else 0\n            )\n            \n            if verbose:\n                print(f\"\\nDistributed simulation complete!\")\n                print(f\"Mean reward per episode: {results.mean_reward:.3f}\")\n                print(f\"Total steps across all workers: {total_steps}\")\n                if results.fairness_index is not None:\n                    print(f\"Fairness index: {results.fairness_index:.3f}\")\n            \n            return results\n            \n        except Exception as e:\n            raise SimulationError(f\"Distributed simulation failed: {str(e)}\")\n    \n    def evaluate_scenarios(self, \n                          scenarios: List[Dict[str, Any]], \n                          metrics: List[str] = None,\n                          aggregate_stats: bool = True) -> Dict[str, SimulationResults]:\n        \"\"\"Evaluate multiple scenarios in parallel.\n        \n        Args:\n            scenarios: List of scenario configurations\n            metrics: Metrics to compute\n            aggregate_stats: Whether to compute aggregate statistics\n            \n        Returns:\n            Dictionary of scenario results\n        \"\"\"\n        try:\n            if metrics is None:\n                metrics = [\"efficiency\", \"fairness\"]\n            \n            # Run scenarios in parallel across workers\n            scenario_futures = []\n            \n            for scenario in scenarios:\n                # Distribute scenario across available workers\n                worker_futures = []\n                for worker in self.workers:\n                    future = worker.evaluate_scenario.remote(scenario, metrics)\n                    worker_futures.append(future)\n                \n                scenario_futures.append(worker_futures)\n            \n            # Collect results\n            scenario_results = {}\n            \n            for i, (scenario, worker_futures) in enumerate(zip(scenarios, scenario_futures)):\n                scenario_name = scenario.get(\"name\", f\"scenario_{i}\")\n                \n                # Get results from all workers for this scenario\n                worker_results = ray.get(worker_futures)\n                \n                # Combine worker results for this scenario\n                combined_rewards = {}\n                combined_stats = {}\n                \n                for worker_result in worker_results:\n                    combined_rewards.update(worker_result.episode_rewards)\n                    combined_stats.update(worker_result.agent_stats)\n                \n                scenario_results[scenario_name] = SimulationResults(\n                    episode_rewards=combined_rewards,\n                    agent_stats=combined_stats,\n                    environment_stats={\"scenario\": scenario}\n                )\n            \n            logger.info(f\"Evaluated {len(scenarios)} scenarios in parallel\")\n            return scenario_results\n            \n        except Exception as e:\n            raise SimulationError(f\"Scenario evaluation failed: {str(e)}\")\n    \n    def shutdown(self) -> None:\n        \"\"\"Shutdown distributed arena and cleanup resources.\"\"\"\n        try:\n            # Shutdown workers\n            if self.workers:\n                shutdown_futures = [worker.shutdown.remote() for worker in self.workers]\n                ray.get(shutdown_futures)\n                self.workers.clear()\n            \n            # Remove placement group\n            if self.placement_group:\n                ray.util.remove_placement_group(self.placement_group)\n                self.placement_group = None\n            \n            logger.info(\"Distributed arena shut down successfully\")\n            \n        except Exception as e:\n            logger.error(f\"Error during distributed arena shutdown: {str(e)}\")\n    \n    def __del__(self) -> None:\n        \"\"\"Cleanup on deletion.\"\"\"\n        try:\n            self.shutdown()\n        except Exception:\n            pass  # Ignore errors during cleanup