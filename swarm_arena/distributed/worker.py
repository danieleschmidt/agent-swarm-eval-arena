"""Ray worker implementation for distributed simulation."""

import ray
import numpy as np
from typing import Dict, List, Any, Optional, Type
import time

from ..core.arena import Arena, SimulationResults
from ..core.config import SwarmConfig  
from ..core.agent import BaseAgent
from ..core.environment import Environment, ForagingEnvironment
from ..exceptions import SimulationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@ray.remote
class ArenaWorker:
    """Ray worker for distributed arena simulation."""
    
    def __init__(self, worker_id: int, config: SwarmConfig, agents_per_worker: int):
        """Initialize arena worker.
        
        Args:
            worker_id: Unique worker identifier
            config: Swarm configuration
            agents_per_worker: Number of agents this worker will handle
        """
        self.worker_id = worker_id
        self.config = config
        self.agents_per_worker = agents_per_worker
        
        # Create worker-specific config
        worker_config = config.copy(num_agents=agents_per_worker)
        self.arena = Arena(worker_config)
        
        logger.info(f"ArenaWorker {worker_id} initialized with {agents_per_worker} agents")
    
    def add_agents(self, agent_class: Type[BaseAgent], count: int) -> str:
        """Add agents to this worker's arena."""
        self.arena.add_agents(agent_class, count)
        return f"Worker {self.worker_id}: Added {count} {agent_class.__name__} agents"
    
    def reset(self) -> Dict[str, Any]:
        """Reset the worker's arena."""
        return self.arena.reset()
    
    def step(self) -> tuple:
        """Execute one simulation step."""
        return self.arena.step()
    
    def run_episode(self, episode_length: Optional[int] = None) -> Dict[str, Any]:
        """Run a complete episode."""
        if episode_length is None:
            episode_length = self.config.episode_length
        
        # Reset arena for new episode
        self.arena.reset()
        
        # Track episode data
        step_count = 0
        total_rewards = {}
        
        # Run episode step by step
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "num_agents": len(self.arena.agents),
            "current_step": self.arena.current_step,
            "arena_size": self.config.arena_size
        }
    
    def shutdown(self) -> None:
        """Shutdown worker."""
        logger.info(f"ArenaWorker {self.worker_id} shutting down")
        # Cleanup resources if needed