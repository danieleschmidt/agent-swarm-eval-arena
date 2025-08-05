"""Parallel and concurrent arena implementation for high performance."""

import asyncio
import concurrent.futures
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

from .arena import Arena, SimulationResults
from .config import SwarmConfig
from ..utils.performance import global_profiler, profile_function, MemoryPool, BatchProcessor
from ..utils.error_handler import with_error_handling
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    
    num_workers: int = 4
    batch_size: int = 50
    enable_async: bool = True
    worker_timeout: float = 30.0
    max_queue_size: int = 1000
    load_balance_threshold: float = 0.8


class WorkerPool:
    """Pool of worker processes for parallel simulation."""
    
    def __init__(self, config: ParallelConfig):
        """Initialize worker pool.
        
        Args:
            config: Parallel processing configuration
        """
        self.config = config
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.num_workers,
            thread_name_prefix="SwarmWorker"
        )
        self.task_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.worker_stats = {i: {"tasks_completed": 0, "avg_time": 0.0} 
                           for i in range(config.num_workers)}
        self.lock = threading.RLock()
        
        logger.info(f"Worker pool initialized with {config.num_workers} workers")
    
    async def submit_batch(self, task_func: Callable, batch_items: List[Any]) -> List[Any]:
        """Submit batch of tasks for parallel processing.
        
        Args:
            task_func: Function to execute on each item
            batch_items: List of items to process
            
        Returns:
            List of results in same order as input
        """
        if not batch_items:
            return []
        
        # Create futures for batch processing
        loop = asyncio.get_event_loop()
        futures = []
        
        for item in batch_items:
            future = loop.run_in_executor(
                self.executor,
                self._execute_with_stats,
                task_func,
                item
            )
            futures.append(future)
        
        # Wait for all tasks to complete
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True),
                timeout=self.config.worker_timeout
            )
            return results
        except asyncio.TimeoutError:
            logger.error("Worker batch timed out")
            return [None] * len(batch_items)
    
    def _execute_with_stats(self, task_func: Callable, item: Any) -> Any:
        """Execute task with performance tracking.
        
        Args:
            task_func: Function to execute
            item: Item to process
            
        Returns:
            Task result
        """
        start_time = time.time()
        try:
            result = task_func(item)
            duration = time.time() - start_time
            
            # Update worker stats
            worker_id = threading.current_thread().ident % self.config.num_workers
            with self.lock:
                stats = self.worker_stats[worker_id]
                stats["tasks_completed"] += 1
                # Running average
                stats["avg_time"] = (stats["avg_time"] * (stats["tasks_completed"] - 1) + duration) / stats["tasks_completed"]
            
            return result
        except Exception as e:
            logger.error(f"Worker task failed: {str(e)}")
            return None
    
    def get_worker_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get worker performance statistics.
        
        Returns:
            Worker statistics dictionary
        """
        with self.lock:
            return dict(self.worker_stats)
    
    def shutdown(self) -> None:
        """Shutdown worker pool."""
        self.executor.shutdown(wait=True)
        logger.info("Worker pool shutdown complete")


class ParallelArena(Arena):
    """High-performance parallel arena implementation."""
    
    def __init__(self, config: SwarmConfig, parallel_config: Optional[ParallelConfig] = None):
        """Initialize parallel arena.
        
        Args:
            config: Swarm configuration
            parallel_config: Parallel processing configuration
        """
        super().__init__(config)
        
        self.parallel_config = parallel_config or ParallelConfig()
        self.worker_pool = WorkerPool(self.parallel_config)
        self.batch_processor = BatchProcessor(
            batch_size=self.parallel_config.batch_size,
            max_wait_time=0.01  # 10ms max wait
        )
        
        # Memory pools for performance
        self.observation_pool = MemoryPool(dict, initial_size=config.num_agents)
        self.action_pool = MemoryPool(dict, initial_size=config.num_agents)
        
        # Performance optimization flags
        self.enable_vectorized_ops = True
        self.enable_batch_processing = True
        self.enable_caching = True
        
        logger.info("Parallel arena initialized with optimizations enabled")
    
    @profile_function("parallel_step")
    async def step_async(self) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, float], bool, Dict[str, Any]]:
        """Execute one simulation step asynchronously with parallel processing.
        
        Returns:
            Tuple of (observations, rewards, done, info)
        """
        step_start_time = time.time()
        
        # Get observations in parallel
        observations = await self._get_observations_parallel()
        
        # Get actions in parallel
        actions = await self._get_actions_parallel(observations)
        
        # Execute physics step (can be parallelized for large numbers of agents)
        if len(self.agents) > 100:
            await self._execute_physics_step_parallel(actions)
        else:
            self._execute_physics_step(actions)
        
        # Execute environment step
        env_obs, env_rewards, env_done, env_info = self.environment.step(actions)
        
        # Calculate rewards in parallel
        rewards = await self._calculate_rewards_parallel(actions, env_rewards)
        
        # Update agent records
        self._update_agent_records(actions, rewards)
        
        # Update simulation state
        self.current_step += 1
        done = (self.current_step >= self.config.episode_length or 
                env_done or 
                not any(agent.state.alive for agent in self.agents.values()))
        
        # Performance tracking
        step_time = time.time() - step_start_time
        self.step_times.append(step_time)
        
        # Memory management
        if len(self.step_times) > self._max_performance_history:
            self.step_times = self.step_times[-self._max_performance_history//2:]
        
        info = {
            "step": self.current_step,
            "active_agents": sum(1 for agent in self.agents.values() if agent.state.alive),
            "step_time": step_time,
            "average_step_time": np.mean(self.step_times[-100:]),
            "parallel_workers": self.parallel_config.num_workers,
            **env_info
        }
        
        return observations, rewards, done, info
    
    async def _get_observations_parallel(self) -> Dict[int, Dict[str, Any]]:
        """Get observations for all agents in parallel.
        
        Returns:
            Dictionary mapping agent_id to observation
        """
        if not self.enable_batch_processing or len(self.agents) < 20:
            return self._get_observations()
        
        # Create batches of agents for parallel processing
        active_agents = [(aid, agent) for aid, agent in self.agents.items() if agent.state.alive]
        
        if not active_agents:
            return {}
        
        # Process observations in parallel batches
        async def process_agent_observation(agent_data):
            agent_id, agent = agent_data
            agent_pos = self.agent_positions[agent_id]
            
            # Use optimized neighbor search
            from ..utils.performance import optimized_neighbor_search
            neighbor_positions = optimized_neighbor_search(
                {aid: pos for aid, pos in self.agent_positions.items() 
                 if aid != agent_id and self.agents[aid].state.alive},
                agent_pos,
                self.config.observation_radius
            )
            
            nearby_agents = [self.agent_positions[aid].tolist() for aid in neighbor_positions]
            
            # Get environment observation
            env_obs = self.environment.get_observation_for_agent(agent_id, agent_pos)
            
            # Create observation
            observation = {
                **env_obs,
                "position": agent_pos.tolist(),
                "velocity": self.agent_velocities[agent_id].tolist(),
                "nearby_agents": nearby_agents,
                "resources": env_obs.get("nearby_resources", []),
                "arena_bounds": {
                    "width": self.config.arena_size[0],
                    "height": self.config.arena_size[1]
                }
            }
            
            return (agent_id, observation)
        
        # Process in parallel
        results = await self.worker_pool.submit_batch(process_agent_observation, active_agents)
        
        # Combine results
        observations = {}
        for result in results:
            if result and len(result) == 2:
                agent_id, obs = result
                observations[agent_id] = obs
        
        return observations
    
    async def _get_actions_parallel(self, observations: Dict[int, Dict[str, Any]]) -> Dict[int, int]:
        """Get actions from all agents in parallel.
        
        Args:
            observations: Agent observations
            
        Returns:
            Dictionary of agent actions
        """
        if not self.enable_batch_processing or len(observations) < 20:
            # Fall back to sequential processing for small numbers
            from ..utils.error_handler import safe_action_execution
            actions = {}
            for agent_id, obs in observations.items():
                if agent_id in self.agents:
                    action = safe_action_execution(self.agents[agent_id], obs)
                    actions[agent_id] = action
            return actions
        
        # Process actions in parallel
        async def process_agent_action(agent_obs_pair):
            agent_id, obs = agent_obs_pair
            if agent_id in self.agents:
                from ..utils.error_handler import safe_action_execution
                action = safe_action_execution(self.agents[agent_id], obs)
                return (agent_id, action)
            return None
        
        # Create list of (agent_id, observation) pairs
        agent_obs_pairs = list(observations.items())
        
        # Process in parallel
        results = await self.worker_pool.submit_batch(process_agent_action, agent_obs_pairs)
        
        # Combine results
        actions = {}
        for result in results:
            if result and len(result) == 2:
                agent_id, action = result
                actions[agent_id] = action
        
        return actions
    
    async def _execute_physics_step_parallel(self, actions: Dict[int, int]) -> None:
        """Execute physics step in parallel for large agent counts.
        
        Args:
            actions: Dictionary of agent actions
        """
        # For very large numbers of agents, we can parallelize physics calculations
        if len(self.agents) < 200:
            self._execute_physics_step(actions)
            return
        
        # Batch physics updates
        async def update_agent_physics(agent_batch):
            results = []
            
            # Convert actions to movement vectors
            action_to_movement = {
                0: np.array([0, 0]),      # no-op
                1: np.array([0, 1]),      # move_up
                2: np.array([0, -1]),     # move_down
                3: np.array([-1, 0]),     # move_left
                4: np.array([1, 0]),      # move_right
                5: np.array([0, 0]),      # collect_resource (no movement)
            }
            
            dt = 1.0 / 60.0  # 60 FPS simulation
            
            for agent_id in agent_batch:
                if agent_id not in actions or not self.agents[agent_id].state.alive:
                    continue
                
                # Update velocity
                action = actions[agent_id]
                movement = action_to_movement.get(action, np.array([0, 0]))
                new_velocity = movement * self.config.max_agent_speed
                
                # Update position
                old_pos = self.agent_positions[agent_id]
                new_pos = old_pos + new_velocity * dt
                
                # Enforce boundaries
                new_pos[0] = np.clip(new_pos[0], 0, self.config.arena_size[0])
                new_pos[1] = np.clip(new_pos[1], 0, self.config.arena_size[1])
                
                results.append((agent_id, new_pos, new_velocity))
            
            return results
        
        # Create batches of agents
        agent_ids = list(self.agents.keys())
        batch_size = max(50, len(agent_ids) // self.parallel_config.num_workers)
        agent_batches = [agent_ids[i:i + batch_size] for i in range(0, len(agent_ids), batch_size)]
        
        # Process batches in parallel
        all_results = await self.worker_pool.submit_batch(update_agent_physics, agent_batches)
        
        # Apply results
        for batch_results in all_results:
            if batch_results:
                for agent_id, new_pos, new_velocity in batch_results:
                    self.agent_positions[agent_id] = new_pos
                    self.agent_velocities[agent_id] = new_velocity
                    self.agents[agent_id].update_state(new_pos, new_velocity)
                    
                    # Update spatial index
                    self.spatial_index.update_agent(agent_id, new_pos)
        
        # Handle collisions if enabled (still sequential for simplicity)
        if self.config.collision_detection:
            self._handle_collisions()
    
    async def _calculate_rewards_parallel(self, actions: Dict[int, int], env_rewards: Dict[int, float]) -> Dict[int, float]:
        """Calculate rewards in parallel.
        
        Args:
            actions: Agent actions
            env_rewards: Environment rewards
            
        Returns:
            Dictionary of total rewards
        """
        if len(actions) < 50:
            return self._calculate_rewards(actions, env_rewards)
        
        # Process rewards in parallel
        async def calculate_agent_reward(agent_action_pair):
            agent_id, action = agent_action_pair
            reward = env_rewards.get(agent_id, 0.0)
            
            # Resource collection reward
            if action == 5 and agent_id in self.agent_positions:  # collect_resource
                collection_reward = self.environment.process_resource_collection(
                    agent_id, self.agent_positions[agent_id]
                )
                reward += collection_reward * self.config.reward_config.get("resource_collection", 1.0)
                
                # Update agent's resource count
                if collection_reward > 0:
                    self.agents[agent_id].state.resources_collected += 1
            
            # Time penalty
            reward += self.config.reward_config.get("time_penalty", -0.001)
            
            # Survival bonus
            if agent_id in self.agents and self.agents[agent_id].state.alive:
                reward += self.config.reward_config.get("survival_bonus", 0.01)
            
            return (agent_id, reward)
        
        # Process in parallel
        results = await self.worker_pool.submit_batch(calculate_agent_reward, list(actions.items()))
        
        # Combine results
        rewards = {}
        for result in results:
            if result and len(result) == 2:
                agent_id, reward = result
                rewards[agent_id] = reward
        
        return rewards
    
    def _update_agent_records(self, actions: Dict[int, int], rewards: Dict[int, float]) -> None:
        """Update agent records efficiently.
        
        Args:
            actions: Agent actions
            rewards: Agent rewards
        """
        for agent_id, agent in self.agents.items():
            if agent_id in actions and agent_id in rewards:
                agent.record_action(actions[agent_id], rewards[agent_id])
                self.episode_rewards[agent_id].append(rewards[agent_id])
    
    async def run_async(self, episodes: int = 1, verbose: bool = True) -> SimulationResults:
        """Run simulation asynchronously with parallel processing.
        
        Args:
            episodes: Number of episodes to run
            verbose: Whether to print progress
            
        Returns:
            Simulation results
        """
        all_episode_rewards = {agent_id: [] for agent_id in self.agents.keys()}
        
        for episode in range(episodes):
            if verbose:
                print(f"Running episode {episode + 1}/{episodes} (parallel mode)")
            
            # Reset for new episode
            self.reset()
            
            # Run episode with async steps
            done = False
            while not done:
                observations, rewards, done, info = await self.step_async()
                
                if verbose and self.current_step % 100 == 0:
                    active_agents = info.get("active_agents", 0)
                    avg_reward = np.mean(list(rewards.values())) if rewards else 0.0
                    step_time = info.get("step_time", 0.0)
                    print(f"  Step {self.current_step}: {active_agents} agents, "
                          f"reward: {avg_reward:.3f}, time: {step_time*1000:.1f}ms")
            
            # Collect episode rewards
            for agent_id in self.agents.keys():
                episode_total = sum(self.episode_rewards[agent_id])
                all_episode_rewards[agent_id].append(episode_total)
        
        # Create results
        results = SimulationResults(
            episode_rewards=all_episode_rewards,
            agent_stats={agent_id: agent.get_stats() for agent_id, agent in self.agents.items()},
            environment_stats=self.environment.get_stats(),
            total_steps=self.current_step * episodes,
            episode_length=self.current_step,
        )
        
        if verbose:
            print(f"\nParallel simulation complete!")
            print(f"Mean reward per episode: {results.mean_reward:.3f}")
            if results.fairness_index is not None:
                print(f"Fairness index: {results.fairness_index:.3f}")
            
            # Performance stats
            perf_stats = global_profiler.get_timing_stats("parallel_step")
            if perf_stats.get("count", 0) > 0:
                print(f"Average step time: {perf_stats['mean']*1000:.1f}ms")
        
        return results
    
    def cleanup(self) -> None:
        """Cleanup parallel arena resources."""
        try:
            self.worker_pool.shutdown()
            super().cleanup() if hasattr(super(), 'cleanup') else None
            logger.info("Parallel arena cleaned up")
        except Exception as e:
            logger.warning(f"Parallel arena cleanup error: {str(e)}")
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass


# Convenience function to create optimized arena
def create_optimized_arena(config: SwarmConfig, 
                          parallel_config: Optional[ParallelConfig] = None) -> ParallelArena:
    """Create an optimized parallel arena.
    
    Args:
        config: Swarm configuration
        parallel_config: Parallel processing configuration
        
    Returns:
        Optimized parallel arena instance
    """
    if parallel_config is None:
        # Auto-configure based on agent count
        num_workers = min(8, max(2, config.num_agents // 50))
        parallel_config = ParallelConfig(
            num_workers=num_workers,
            batch_size=max(10, config.num_agents // num_workers),
            enable_async=True
        )
    
    arena = ParallelArena(config, parallel_config)
    
    logger.info(f"Created optimized arena with {parallel_config.num_workers} workers")
    return arena