"""Ray-based distributed arena for large-scale simulations."""

import ray
import numpy as np
from typing import Dict, List, Any, Optional, Type, Tuple
import time
from dataclasses import dataclass

from ..core.arena import Arena, SimulationResults
from ..core.config import SwarmConfig
from ..core.agent import BaseAgent
from ..core.environment import ForagingEnvironment
from ..exceptions import SimulationError
from ..utils.logging import get_logger

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


@ray.remote
class DistributedWorker:
    """Ray actor for distributed arena execution."""
    
    def __init__(self, config: SwarmConfig, worker_id: int, agents_per_worker: int):
        """Initialize distributed worker."""
        self.worker_id = worker_id
        self.agents_per_worker = agents_per_worker
        
        # Create local arena
        worker_config = config.copy(num_agents=agents_per_worker)
        self.arena = Arena(worker_config)
        
        logger.info(f"Distributed worker {worker_id} initialized with {agents_per_worker} agents")
    
    def add_agents(self, agent_class: Type[BaseAgent], count: int):
        """Add agents to this worker."""
        self.arena.add_agents(agent_class, count)
        return f"Worker {self.worker_id}: Added {count} {agent_class.__name__} agents"
    
    def reset(self):
        """Reset the worker arena."""
        return self.arena.reset()
    
    def step(self):
        """Execute one step on worker."""
        return self.arena.step()
    
    def run_episode(self, episode_length: int):
        """Run a complete episode on worker."""
        self.arena.reset()
        
        step_count = 0
        total_rewards = {}
        
        for step in range(episode_length):
            observations, rewards, done, info = self.arena.step()
            
            # Accumulate rewards
            for agent_id, reward in rewards.items():
                if agent_id not in total_rewards:
                    total_rewards[agent_id] = []
                total_rewards[agent_id].append(reward)
            
            step_count += 1
            if done:
                break
        
        return {
            "worker_id": self.worker_id,
            "steps": step_count,
            "total_rewards": total_rewards,
            "agent_stats": {aid: agent.get_stats() for aid, agent in self.arena.agents.items()}
        }


class DistributedArena:
    """Distributed arena using Ray for massive scale simulations."""
    
    def __init__(self, 
                 config: SwarmConfig,
                 distributed_config: Optional[DistributedConfig] = None,
                 environment: Optional[Any] = None) -> None:
        """Initialize distributed arena."""
        self.config = config
        self.distributed_config = distributed_config or DistributedConfig()
        
        # Initialize Ray if needed
        if not ray.is_initialized():
            try:
                ray.init(ignore_reinit_error=True)
                logger.info("Ray initialized for distributed arena")
            except Exception as e:
                raise SimulationError(f"Failed to initialize Ray: {str(e)}")
        
        # Create distributed workers
        self.workers = []
        agents_per_worker = self.distributed_config.agents_per_worker
        
        for i in range(self.distributed_config.num_workers):
            worker = DistributedWorker.remote(config, i, agents_per_worker)
            self.workers.append(worker)
        
        logger.info(f"Created {len(self.workers)} distributed workers")
        
        # Track agent assignments
        self.agent_assignments = {}
        self.total_agents_added = 0
    
    def add_agents(self, agent_class: Type[BaseAgent], count: int) -> None:
        """Add agents distributed across workers."""
        agents_per_worker = count // len(self.workers)
        remaining_agents = count % len(self.workers)
        
        futures = []
        for i, worker in enumerate(self.workers):
            worker_agents = agents_per_worker
            if i < remaining_agents:
                worker_agents += 1
            
            if worker_agents > 0:
                future = worker.add_agents.remote(agent_class, worker_agents)
                futures.append(future)
                
                # Track assignment
                start_id = self.total_agents_added
                for j in range(worker_agents):
                    self.agent_assignments[start_id + j] = i
                self.total_agents_added += worker_agents
        
        # Wait for completion
        ray.get(futures)
        logger.info(f"Distributed {count} {agent_class.__name__} agents across {len(self.workers)} workers")
    
    def reset(self) -> Dict[str, Any]:
        """Reset all distributed workers."""
        reset_futures = [worker.reset.remote() for worker in self.workers]
        reset_results = ray.get(reset_futures)
        
        logger.info("Reset all distributed workers")
        return {
            "num_workers": len(self.workers),
            "total_agents": self.total_agents_added,
            "worker_results": reset_results
        }
    
    def run(self, episodes: int = 1, verbose: bool = True) -> SimulationResults:
        """Run distributed simulation."""
        logger.info(f"Starting distributed simulation with {episodes} episodes")
        
        all_episode_rewards = {}
        all_agent_stats = {}
        total_steps = 0
        
        for episode in range(episodes):
            if verbose:
                print(f"Running distributed episode {episode + 1}/{episodes}")
            
            # Run episode on all workers in parallel
            episode_futures = [
                worker.run_episode.remote(self.config.episode_length) 
                for worker in self.workers
            ]
            
            # Collect results from all workers
            worker_results = ray.get(episode_futures)
            
            # Aggregate results
            episode_rewards = {}
            episode_stats = {}
            
            for result in worker_results:
                worker_id = result["worker_id"]
                total_steps += result["steps"]
                
                # Merge rewards (with worker offset for agent IDs)
                base_agent_id = worker_id * self.distributed_config.agents_per_worker
                for local_agent_id, rewards in result["total_rewards"].items():
                    global_agent_id = base_agent_id + local_agent_id
                    episode_rewards[global_agent_id] = rewards
                
                # Merge agent stats
                for local_agent_id, stats in result["agent_stats"].items():
                    global_agent_id = base_agent_id + local_agent_id
                    episode_stats[global_agent_id] = stats
            
            # Store episode results
            for agent_id, rewards in episode_rewards.items():
                if agent_id not in all_episode_rewards:
                    all_episode_rewards[agent_id] = []
                all_episode_rewards[agent_id].extend(rewards)
            
            # Update agent stats
            all_agent_stats.update(episode_stats)
            
            if verbose:
                total_agents = len(episode_rewards)
                avg_reward = np.mean([sum(rewards) for rewards in episode_rewards.values()]) if episode_rewards else 0
                print(f"  Episode {episode + 1}: {total_agents} agents, avg reward: {avg_reward:.3f}")
        
        # Create results
        results = SimulationResults(
            episode_rewards=all_episode_rewards,
            agent_stats=all_agent_stats,
            environment_stats={"distributed_workers": len(self.workers)},
            total_steps=total_steps,
            episode_length=self.config.episode_length
        )
        
        if verbose:
            print(f"Distributed simulation complete!")
            print(f"Mean reward per episode: {results.mean_reward:.3f}")
            if results.fairness_index is not None:
                print(f"Fairness index: {results.fairness_index:.3f}")
        
        return results
    
    def evaluate_scenarios(self, scenarios: List[Dict], metrics: List[str] = None) -> Dict[str, Any]:
        """Evaluate different scenarios in parallel."""
        logger.info(f"Evaluating {len(scenarios)} scenarios")
        
        scenario_results = {}
        
        for scenario in scenarios:
            scenario_name = scenario.get("name", "unnamed_scenario")
            logger.info(f"Running scenario: {scenario_name}")
            
            # For this simplified implementation, just run with current setup
            # In a full implementation, this would modify worker configurations
            results = self.run(episodes=1, verbose=False)
            scenario_results[scenario_name] = results
        
        return scenario_results
    
    def shutdown(self) -> None:
        """Shutdown distributed arena."""
        try:
            # Clean up workers
            for worker in self.workers:
                ray.kill(worker)
            
            self.workers.clear()
            logger.info("Distributed arena shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during distributed arena shutdown: {str(e)}")
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup