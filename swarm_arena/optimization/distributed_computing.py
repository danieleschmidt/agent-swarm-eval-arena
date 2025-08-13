"""Distributed computing infrastructure for large-scale simulations."""

import time
import pickle
import threading
import queue
import socket
import uuid
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import ray
from concurrent.futures import ThreadPoolExecutor
import redis


class NodeRole(Enum):
    """Roles for distributed nodes."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    STORAGE = "storage"
    MONITOR = "monitor"


class TaskStatus(Enum):
    """Status of distributed tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NodeInfo:
    """Information about a distributed node."""
    node_id: str
    role: NodeRole
    address: str
    port: int
    capabilities: Dict[str, Any] = field(default_factory=dict)
    load: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "active"


@dataclass
class Task:
    """Distributed task definition."""
    task_id: str
    function_name: str
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class TaskResult:
    """Result of a distributed task."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    completed_at: float = field(default_factory=time.time)
    worker_id: Optional[str] = None


class DistributedStorage:
    """Distributed storage interface using Redis."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize distributed storage.
        
        Args:
            redis_url: Redis connection URL
        """
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_available = True
        except:
            self.redis_client = None
            self.redis_available = False
            logging.warning("Redis not available, using local storage")
        
        # Fallback to local storage
        self.local_storage: Dict[str, Any] = {}
        self.lock = threading.Lock()
    
    def store(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value in distributed storage.
        
        Args:
            key: Storage key
            value: Value to store
            ttl: Time to live in seconds
            
        Returns:
            True if stored successfully
        """
        try:
            if self.redis_available:
                serialized = pickle.dumps(value)
                if ttl:
                    return self.redis_client.setex(key, ttl, serialized)
                else:
                    return self.redis_client.set(key, serialized)
            else:
                with self.lock:
                    self.local_storage[key] = value
                    return True
        except Exception as e:
            logging.error(f"Storage error: {e}")
            return False
    
    def retrieve(self, key: str) -> Any:
        """Retrieve a value from distributed storage.
        
        Args:
            key: Storage key
            
        Returns:
            Stored value or None if not found
        """
        try:
            if self.redis_available:
                serialized = self.redis_client.get(key)
                if serialized:
                    return pickle.loads(serialized)
                return None
            else:
                with self.lock:
                    return self.local_storage.get(key)
        except Exception as e:
            logging.error(f"Retrieval error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a value from storage.
        
        Args:
            key: Storage key
            
        Returns:
            True if deleted successfully
        """
        try:
            if self.redis_available:
                return bool(self.redis_client.delete(key))
            else:
                with self.lock:
                    if key in self.local_storage:
                        del self.local_storage[key]
                        return True
                    return False
        except Exception as e:
            logging.error(f"Deletion error: {e}")
            return False
    
    def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern.
        
        Args:
            pattern: Key pattern
            
        Returns:
            List of matching keys
        """
        try:
            if self.redis_available:
                return [key.decode() for key in self.redis_client.keys(pattern)]
            else:
                with self.lock:
                    import fnmatch
                    return [key for key in self.local_storage.keys() 
                           if fnmatch.fnmatch(key, pattern)]
        except Exception as e:
            logging.error(f"List keys error: {e}")
            return []


class NodeManager:
    """Manages distributed nodes and their communication."""
    
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or str(uuid.uuid4())
        self.nodes: Dict[str, NodeInfo] = {}
        self.storage = DistributedStorage()
        self.lock = threading.Lock()
        
        # Heartbeat monitoring
        self.heartbeat_interval = 30.0  # seconds
        self.node_timeout = 90.0  # seconds
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        
        self.logger = logging.getLogger(f'node_manager.{self.node_id}')
    
    def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new node.
        
        Args:
            node_info: Node information
            
        Returns:
            True if registered successfully
        """
        with self.lock:
            self.nodes[node_info.node_id] = node_info
        
        # Store in distributed storage
        key = f"node:{node_info.node_id}"
        success = self.storage.store(key, node_info.__dict__, ttl=int(self.node_timeout))
        
        self.logger.info(f"Registered node {node_info.node_id} with role {node_info.role.value}")
        return success
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a node.
        
        Args:
            node_id: Node ID to unregister
            
        Returns:
            True if unregistered successfully
        """
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
        
        key = f"node:{node_id}"
        success = self.storage.delete(key)
        
        self.logger.info(f"Unregistered node {node_id}")
        return success
    
    def get_nodes_by_role(self, role: NodeRole) -> List[NodeInfo]:
        """Get all nodes with specific role.
        
        Args:
            role: Node role to filter by
            
        Returns:
            List of nodes with the specified role
        """
        with self.lock:
            return [node for node in self.nodes.values() if node.role == role]
    
    def update_node_load(self, node_id: str, load: float) -> None:
        """Update node load information.
        
        Args:
            node_id: Node ID
            load: Current load (0.0 to 1.0)
        """
        with self.lock:
            if node_id in self.nodes:
                self.nodes[node_id].load = load
                self.nodes[node_id].last_heartbeat = time.time()
    
    def get_least_loaded_worker(self) -> Optional[NodeInfo]:
        """Get the worker node with lowest load.
        
        Returns:
            Least loaded worker node or None
        """
        workers = self.get_nodes_by_role(NodeRole.WORKER)
        if not workers:
            return None
        
        # Filter active workers and sort by load
        active_workers = [
            worker for worker in workers
            if (time.time() - worker.last_heartbeat) < self.node_timeout
        ]
        
        if not active_workers:
            return None
        
        return min(active_workers, key=lambda w: w.load)
    
    def start_monitoring(self) -> None:
        """Start node monitoring and cleanup."""
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Node monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop node monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Node monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Monitor nodes and clean up inactive ones."""
        while self.running:
            try:
                current_time = time.time()
                inactive_nodes = []
                
                with self.lock:
                    for node_id, node in self.nodes.items():
                        if current_time - node.last_heartbeat > self.node_timeout:
                            inactive_nodes.append(node_id)
                
                # Remove inactive nodes
                for node_id in inactive_nodes:
                    self.unregister_node(node_id)
                    self.logger.warning(f"Removed inactive node: {node_id}")
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.heartbeat_interval)


class TaskScheduler:
    """Distributes and schedules tasks across worker nodes."""
    
    def __init__(self, node_manager: NodeManager):
        self.node_manager = node_manager
        self.storage = node_manager.storage
        
        # Task queues
        self.pending_tasks: queue.PriorityQueue = queue.PriorityQueue()
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        
        # Task execution
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        
        self.logger = logging.getLogger('task_scheduler')
    
    def submit_task(self, task: Task) -> str:
        """Submit a task for execution.
        
        Args:
            task: Task to execute
            
        Returns:
            Task ID
        """
        # Store task in distributed storage
        key = f"task:{task.task_id}"
        self.storage.store(key, task.__dict__)
        
        # Add to pending queue (negative priority for min-heap behavior)
        self.pending_tasks.put((-task.priority, task.created_at, task))
        
        self.logger.info(f"Submitted task {task.task_id}")
        return task.task_id
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of a completed task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result or None if not completed
        """
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        # Try to load from storage
        key = f"result:{task_id}"
        result_data = self.storage.retrieve(key)
        
        if result_data:
            result = TaskResult(**result_data)
            self.completed_tasks[task_id] = result
            return result
        
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        # Remove from running tasks
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]
        
        # Mark as cancelled in storage
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.CANCELLED,
            completed_at=time.time()
        )
        
        key = f"result:{task_id}"
        success = self.storage.store(key, result.__dict__)
        
        if success:
            self.completed_tasks[task_id] = result
        
        return success
    
    def start_scheduling(self) -> None:
        """Start task scheduling."""
        if self.running:
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduling_loop, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("Task scheduling started")
    
    def stop_scheduling(self) -> None:
        """Stop task scheduling."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        self.logger.info("Task scheduling stopped")
    
    def _scheduling_loop(self) -> None:
        """Main scheduling loop."""
        while self.running:
            try:
                # Get next task from queue
                try:
                    _, _, task = self.pending_tasks.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check dependencies
                if not self._dependencies_satisfied(task):
                    # Re-queue task
                    self.pending_tasks.put((-task.priority, task.created_at, task))
                    time.sleep(0.1)
                    continue
                
                # Find worker for task
                worker = self.node_manager.get_least_loaded_worker()
                
                if worker is None:
                    # No workers available, re-queue
                    self.pending_tasks.put((-task.priority, task.created_at, task))
                    time.sleep(1.0)
                    continue
                
                # Execute task
                self.running_tasks[task.task_id] = task
                future = self.executor.submit(self._execute_task, task, worker)
                
                self.logger.info(f"Scheduled task {task.task_id} on worker {worker.node_id}")
                
            except Exception as e:
                self.logger.error(f"Error in scheduling loop: {e}")
                time.sleep(1.0)
    
    def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if task dependencies are satisfied.
        
        Args:
            task: Task to check
            
        Returns:
            True if all dependencies are completed
        """
        for dep_id in task.dependencies:
            result = self.get_task_result(dep_id)
            if not result or result.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def _execute_task(self, task: Task, worker: NodeInfo) -> None:
        """Execute a task on a worker.
        
        Args:
            task: Task to execute
            worker: Worker node to execute on
        """
        start_time = time.time()
        
        try:
            # Update task status
            key = f"task:{task.task_id}"
            task_data = task.__dict__.copy()
            task_data['status'] = TaskStatus.RUNNING.value
            task_data['worker_id'] = worker.node_id
            self.storage.store(key, task_data)
            
            # Execute task (simplified - would use actual RPC in production)
            result = self._simulate_task_execution(task)
            
            # Create result
            task_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=time.time() - start_time,
                worker_id=worker.node_id
            )
            
            self.logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            # Handle task failure
            task_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
                worker_id=worker.node_id
            )
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
            # Retry if possible
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                self.pending_tasks.put((-task.priority, time.time(), task))
                return
        
        finally:
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
        
        # Store result
        result_key = f"result:{task.task_id}"
        self.storage.store(result_key, task_result.__dict__)
        self.completed_tasks[task.task_id] = task_result
    
    def _simulate_task_execution(self, task: Task) -> Any:
        """Simulate task execution (placeholder).
        
        Args:
            task: Task to execute
            
        Returns:
            Simulated result
        """
        # In a real implementation, this would dispatch to the actual function
        time.sleep(0.1)  # Simulate work
        return f"Result for {task.function_name} with args {task.args}"
    
    def get_scheduling_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        return {
            'pending_tasks': self.pending_tasks.qsize(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'workers_available': len(self.node_manager.get_nodes_by_role(NodeRole.WORKER))
        }


class DistributedSimulationManager:
    """High-level manager for distributed simulations."""
    
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or str(uuid.uuid4())
        self.node_manager = NodeManager(self.node_id)
        self.task_scheduler = TaskScheduler(self.node_manager)
        
        # Ray integration
        self.ray_initialized = False
        
        self.logger = logging.getLogger(f'distributed_manager.{self.node_id}')
    
    def initialize_cluster(self, 
                          ray_address: Optional[str] = None,
                          num_cpus: Optional[int] = None) -> bool:
        """Initialize Ray cluster for distributed computing.
        
        Args:
            ray_address: Ray cluster address (None for local)
            num_cpus: Number of CPUs to use
            
        Returns:
            True if initialized successfully
        """
        try:
            if ray_address:
                ray.init(address=ray_address)
            else:
                ray.init(num_cpus=num_cpus)
            
            self.ray_initialized = True
            self.logger.info("Ray cluster initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ray: {e}")
            return False
    
    def start(self) -> None:
        """Start the distributed simulation manager."""
        # Start node monitoring
        self.node_manager.start_monitoring()
        
        # Start task scheduling
        self.task_scheduler.start_scheduling()
        
        # Register this node as coordinator
        coordinator_info = NodeInfo(
            node_id=self.node_id,
            role=NodeRole.COORDINATOR,
            address="localhost",
            port=8000,
            capabilities={"coordination": True, "scheduling": True}
        )
        
        self.node_manager.register_node(coordinator_info)
        
        self.logger.info("Distributed simulation manager started")
    
    def stop(self) -> None:
        """Stop the distributed simulation manager."""
        self.task_scheduler.stop_scheduling()
        self.node_manager.stop_monitoring()
        
        if self.ray_initialized:
            ray.shutdown()
        
        self.logger.info("Distributed simulation manager stopped")
    
    def submit_simulation_task(self, 
                             simulation_func: Callable,
                             *args, 
                             priority: int = 1,
                             **kwargs) -> str:
        """Submit a simulation task for distributed execution.
        
        Args:
            simulation_func: Simulation function to execute
            *args: Function arguments
            priority: Task priority
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        task = Task(
            task_id=str(uuid.uuid4()),
            function_name=simulation_func.__name__,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        return self.task_scheduler.submit_task(task)
    
    def get_simulation_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of a simulation task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result or None if not completed
        """
        return self.task_scheduler.get_task_result(task_id)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status information."""
        return {
            'node_id': self.node_id,
            'nodes': len(self.node_manager.nodes),
            'workers': len(self.node_manager.get_nodes_by_role(NodeRole.WORKER)),
            'scheduling_stats': self.task_scheduler.get_scheduling_stats(),
            'ray_initialized': self.ray_initialized
        }


# Ray remote functions for distributed computing

@ray.remote
class DistributedArena:
    """Ray remote arena for distributed simulation."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        from ..core.arena import Arena
        from ..core.config import SwarmConfig
        
        config = SwarmConfig(**config_dict)
        self.arena = Arena(config)
        self.worker_id = str(uuid.uuid4())
    
    def add_agents(self, agent_class_name: str, count: int):
        """Add agents to the arena."""
        # Import agent class dynamically
        from ..core.agent import CooperativeAgent, CompetitiveAgent, RandomAgent
        
        agent_classes = {
            'CooperativeAgent': CooperativeAgent,
            'CompetitiveAgent': CompetitiveAgent,
            'RandomAgent': RandomAgent
        }
        
        agent_class = agent_classes.get(agent_class_name, RandomAgent)
        self.arena.add_agents(agent_class, count)
    
    def run_episodes(self, episodes: int) -> Dict[str, Any]:
        """Run simulation episodes."""
        results = self.arena.run(episodes=episodes, verbose=False)
        
        return {
            'mean_reward': results.mean_reward,
            'fairness_index': results.fairness_index,
            'total_steps': results.total_steps,
            'worker_id': self.worker_id
        }


def create_distributed_simulation(num_arenas: int = 4,
                                config_dict: Optional[Dict[str, Any]] = None) -> List[ray.ObjectRef]:
    """Create distributed simulation across multiple arenas.
    
    Args:
        num_arenas: Number of parallel arenas
        config_dict: Arena configuration
        
    Returns:
        List of Ray object references
    """
    if config_dict is None:
        config_dict = {
            'num_agents': 50,
            'arena_size': [500, 500],
            'episode_length': 100,
            'seed': 42
        }
    
    # Create distributed arenas
    arenas = [DistributedArena.remote(config_dict) for _ in range(num_arenas)]
    
    # Add agents to each arena
    for arena in arenas:
        arena.add_agents.remote('CooperativeAgent', 25)
        arena.add_agents.remote('CompetitiveAgent', 25)
    
    # Start simulations
    futures = [arena.run_episodes.remote(10) for arena in arenas]
    
    return futures


# Global distributed manager
distributed_manager = DistributedSimulationManager()