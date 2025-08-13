"""Auto-scaling and load balancing for swarm simulations."""

import time
import threading
import queue
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future


class ScalingDirection(Enum):
    """Direction of scaling."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class WorkerState(Enum):
    """State of a worker."""
    IDLE = "idle"
    BUSY = "busy"
    STARTING = "starting"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    queue_length: int = 0
    active_workers: int = 0
    avg_response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingPolicy:
    """Policy for auto-scaling decisions."""
    min_workers: int = 1
    max_workers: int = 10
    target_cpu_usage: float = 0.7
    target_queue_length: int = 5
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.5
    cooldown_period: float = 60.0  # seconds
    evaluation_period: float = 30.0  # seconds
    scale_factor: float = 1.5


@dataclass
class WorkerInfo:
    """Information about a worker."""
    worker_id: str
    state: WorkerState
    created_at: float
    last_activity: float
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_task: Optional[str] = None


class LoadBalancer:
    """Load balancer for distributing work across workers."""
    
    def __init__(self):
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_queues: Dict[str, queue.Queue] = {}
        self.round_robin_index = 0
        self.lock = threading.Lock()
        
        # Load balancing strategies
        self.strategies = {
            'round_robin': self._round_robin_strategy,
            'least_loaded': self._least_loaded_strategy,
            'weighted_round_robin': self._weighted_round_robin_strategy,
            'random': self._random_strategy
        }
        
        self.current_strategy = 'least_loaded'
    
    def add_worker(self, worker_id: str) -> None:
        """Add a worker to the load balancer.
        
        Args:
            worker_id: Unique worker identifier
        """
        with self.lock:
            self.workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                state=WorkerState.IDLE,
                created_at=time.time(),
                last_activity=time.time()
            )
            self.worker_queues[worker_id] = queue.Queue()
    
    def remove_worker(self, worker_id: str) -> None:
        """Remove a worker from the load balancer.
        
        Args:
            worker_id: Worker identifier to remove
        """
        with self.lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
            if worker_id in self.worker_queues:
                del self.worker_queues[worker_id]
    
    def get_worker_for_task(self, task_size: float = 1.0) -> Optional[str]:
        """Get the best worker for a task based on current strategy.
        
        Args:
            task_size: Relative size/complexity of the task
            
        Returns:
            Worker ID or None if no workers available
        """
        with self.lock:
            available_workers = [
                worker_id for worker_id, worker in self.workers.items()
                if worker.state in [WorkerState.IDLE, WorkerState.BUSY]
            ]
            
            if not available_workers:
                return None
            
            strategy_func = self.strategies.get(self.current_strategy, self._least_loaded_strategy)
            return strategy_func(available_workers, task_size)
    
    def _round_robin_strategy(self, workers: List[str], task_size: float) -> str:
        """Round-robin worker selection."""
        if not workers:
            return None
        
        worker = workers[self.round_robin_index % len(workers)]
        self.round_robin_index += 1
        return worker
    
    def _least_loaded_strategy(self, workers: List[str], task_size: float) -> str:
        """Select worker with smallest queue."""
        if not workers:
            return None
        
        worker_loads = [
            (worker_id, self.worker_queues[worker_id].qsize())
            for worker_id in workers
        ]
        
        # Sort by queue size
        worker_loads.sort(key=lambda x: x[1])
        return worker_loads[0][0]
    
    def _weighted_round_robin_strategy(self, workers: List[str], task_size: float) -> str:
        """Weighted round-robin based on worker performance."""
        if not workers:
            return None
        
        # Calculate weights based on success rate and speed
        worker_weights = []
        for worker_id in workers:
            worker = self.workers[worker_id]
            total_tasks = worker.tasks_completed + worker.tasks_failed
            
            if total_tasks > 0:
                success_rate = worker.tasks_completed / total_tasks
                # Weight by success rate and recent activity
                weight = success_rate * (1.0 / (time.time() - worker.last_activity + 1))
            else:
                weight = 1.0
            
            worker_weights.append((worker_id, weight))
        
        # Select based on weights
        total_weight = sum(weight for _, weight in worker_weights)
        if total_weight == 0:
            return workers[0]
        
        # Weighted random selection
        import random
        target = random.random() * total_weight
        current = 0
        
        for worker_id, weight in worker_weights:
            current += weight
            if current >= target:
                return worker_id
        
        return workers[-1]
    
    def _random_strategy(self, workers: List[str], task_size: float) -> str:
        """Random worker selection."""
        import random
        return random.choice(workers) if workers else None
    
    def update_worker_state(self, worker_id: str, state: WorkerState, 
                          task_result: Optional[Dict[str, Any]] = None) -> None:
        """Update worker state and statistics.
        
        Args:
            worker_id: Worker identifier
            state: New worker state
            task_result: Result of completed task (optional)
        """
        with self.lock:
            if worker_id not in self.workers:
                return
            
            worker = self.workers[worker_id]
            worker.state = state
            worker.last_activity = time.time()
            
            if task_result:
                if task_result.get('success', False):
                    worker.tasks_completed += 1
                else:
                    worker.tasks_failed += 1
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self.lock:
            total_workers = len(self.workers)
            active_workers = sum(
                1 for worker in self.workers.values()
                if worker.state in [WorkerState.IDLE, WorkerState.BUSY]
            )
            
            total_queue_size = sum(q.qsize() for q in self.worker_queues.values())
            
            worker_stats = {}
            for worker_id, worker in self.workers.items():
                worker_stats[worker_id] = {
                    'state': worker.state.value,
                    'queue_size': self.worker_queues[worker_id].qsize(),
                    'tasks_completed': worker.tasks_completed,
                    'tasks_failed': worker.tasks_failed,
                    'success_rate': (
                        worker.tasks_completed / (worker.tasks_completed + worker.tasks_failed)
                        if (worker.tasks_completed + worker.tasks_failed) > 0 else 0.0
                    )
                }
            
            return {
                'total_workers': total_workers,
                'active_workers': active_workers,
                'total_queue_size': total_queue_size,
                'strategy': self.current_strategy,
                'worker_stats': worker_stats
            }


class AutoScaler:
    """Automatic scaling based on system metrics and policies."""
    
    def __init__(self, policy: ScalingPolicy, load_balancer: LoadBalancer):
        self.policy = policy
        self.load_balancer = load_balancer
        
        # Metrics history
        self.metrics_history: List[ScalingMetrics] = []
        self.max_history = 100
        
        # Scaling state
        self.last_scaling_action: Optional[float] = None
        self.scaling_decisions: List[Dict[str, Any]] = []
        
        # Monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        
        self.logger = logging.getLogger('auto_scaler')
    
    def start_monitoring(self) -> None:
        """Start automatic monitoring and scaling."""
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop automatic monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Auto-scaling monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep history bounded
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history//2:]
                
                # Make scaling decision
                decision = self._make_scaling_decision(metrics)
                
                if decision['action'] != ScalingDirection.NONE:
                    self._execute_scaling_decision(decision)
                
                time.sleep(self.policy.evaluation_period)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.policy.evaluation_period)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
        memory = psutil.virtual_memory()
        memory_usage = memory.percent / 100.0
        
        # Load balancer metrics
        lb_stats = self.load_balancer.get_load_statistics()
        
        # Calculate response time and throughput from recent history
        avg_response_time = 0.0
        throughput = 0.0
        
        if len(self.metrics_history) > 1:
            recent_metrics = self.metrics_history[-5:]  # Last 5 measurements
            response_times = [m.avg_response_time for m in recent_metrics if m.avg_response_time > 0]
            if response_times:
                avg_response_time = np.mean(response_times)
            
            # Calculate throughput as tasks/second
            time_window = self.policy.evaluation_period * len(recent_metrics)
            total_tasks = sum(
                sum(worker['tasks_completed'] for worker in lb_stats['worker_stats'].values())
                for _ in recent_metrics
            )
            throughput = total_tasks / time_window if time_window > 0 else 0.0
        
        return ScalingMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            queue_length=lb_stats['total_queue_size'],
            active_workers=lb_stats['active_workers'],
            avg_response_time=avg_response_time,
            throughput=throughput,
            error_rate=0.0  # Would be calculated from task results
        )
    
    def _make_scaling_decision(self, current_metrics: ScalingMetrics) -> Dict[str, Any]:
        """Make scaling decision based on metrics and policy.
        
        Args:
            current_metrics: Current system metrics
            
        Returns:
            Scaling decision dictionary
        """
        decision = {
            'action': ScalingDirection.NONE,
            'target_workers': current_metrics.active_workers,
            'reason': 'No action needed',
            'metrics': current_metrics,
            'timestamp': time.time()
        }
        
        # Check cooldown period
        if (self.last_scaling_action and 
            time.time() - self.last_scaling_action < self.policy.cooldown_period):
            decision['reason'] = 'In cooldown period'
            return decision
        
        # Scale up conditions
        scale_up_needed = any([
            current_metrics.cpu_usage > self.policy.scale_up_threshold,
            current_metrics.queue_length > self.policy.target_queue_length * 2,
            current_metrics.avg_response_time > 5.0  # 5 second response time threshold
        ])
        
        # Scale down conditions
        scale_down_needed = all([
            current_metrics.cpu_usage < self.policy.scale_down_threshold,
            current_metrics.queue_length < self.policy.target_queue_length // 2,
            current_metrics.avg_response_time < 1.0,  # 1 second response time
            current_metrics.active_workers > self.policy.min_workers
        ])
        
        if scale_up_needed and current_metrics.active_workers < self.policy.max_workers:
            # Calculate target workers
            target_workers = min(
                self.policy.max_workers,
                max(
                    current_metrics.active_workers + 1,
                    int(current_metrics.active_workers * self.policy.scale_factor)
                )
            )
            
            decision.update({
                'action': ScalingDirection.UP,
                'target_workers': target_workers,
                'reason': f'Scaling up: CPU={current_metrics.cpu_usage:.2f}, Queue={current_metrics.queue_length}'
            })
        
        elif scale_down_needed:
            # Calculate target workers
            target_workers = max(
                self.policy.min_workers,
                int(current_metrics.active_workers / self.policy.scale_factor)
            )
            
            decision.update({
                'action': ScalingDirection.DOWN,
                'target_workers': target_workers,
                'reason': f'Scaling down: CPU={current_metrics.cpu_usage:.2f}, Queue={current_metrics.queue_length}'
            })
        
        return decision
    
    def _execute_scaling_decision(self, decision: Dict[str, Any]) -> None:
        """Execute a scaling decision.
        
        Args:
            decision: Scaling decision to execute
        """
        current_workers = decision['metrics'].active_workers
        target_workers = decision['target_workers']
        
        self.logger.info(f"Executing scaling decision: {decision['action'].value} from {current_workers} to {target_workers} workers")
        
        if decision['action'] == ScalingDirection.UP:
            workers_to_add = target_workers - current_workers
            for i in range(workers_to_add):
                worker_id = f"worker_{int(time.time())}_{i}"
                self.load_balancer.add_worker(worker_id)
                self.logger.info(f"Added worker: {worker_id}")
        
        elif decision['action'] == ScalingDirection.DOWN:
            workers_to_remove = current_workers - target_workers
            
            # Get least active workers to remove
            lb_stats = self.load_balancer.get_load_statistics()
            worker_activity = [
                (worker_id, stats['tasks_completed'] + stats['queue_size'])
                for worker_id, stats in lb_stats['worker_stats'].items()
            ]
            
            # Sort by activity (least active first)
            worker_activity.sort(key=lambda x: x[1])
            
            for i in range(min(workers_to_remove, len(worker_activity))):
                worker_id = worker_activity[i][0]
                self.load_balancer.remove_worker(worker_id)
                self.logger.info(f"Removed worker: {worker_id}")
        
        self.last_scaling_action = time.time()
        self.scaling_decisions.append(decision)
        
        # Keep decision history bounded
        if len(self.scaling_decisions) > 100:
            self.scaling_decisions = self.scaling_decisions[-50:]
    
    def get_scaling_history(self) -> List[Dict[str, Any]]:
        """Get history of scaling decisions."""
        return self.scaling_decisions.copy()
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status."""
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'running': self.running,
            'policy': {
                'min_workers': self.policy.min_workers,
                'max_workers': self.policy.max_workers,
                'target_cpu_usage': self.policy.target_cpu_usage,
                'cooldown_period': self.policy.cooldown_period
            },
            'current_metrics': current_metrics.__dict__ if current_metrics else None,
            'last_scaling_action': self.last_scaling_action,
            'recent_decisions': self.scaling_decisions[-5:] if self.scaling_decisions else []
        }


class ScalingManager:
    """High-level manager for auto-scaling functionality."""
    
    def __init__(self, policy: Optional[ScalingPolicy] = None):
        self.policy = policy or ScalingPolicy()
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler(self.policy, self.load_balancer)
        
        # Task execution
        self.task_queue: queue.Queue = queue.Queue()
        self.result_futures: Dict[str, Future] = {}
        
        # Performance tracking
        self.performance_stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_execution_time': 0.0,
            'throughput': 0.0
        }
    
    def start(self) -> None:
        """Start the scaling manager."""
        # Add initial workers
        for i in range(self.policy.min_workers):
            worker_id = f"worker_init_{i}"
            self.load_balancer.add_worker(worker_id)
        
        # Start auto-scaling
        self.auto_scaler.start_monitoring()
    
    def stop(self) -> None:
        """Stop the scaling manager."""
        self.auto_scaler.stop_monitoring()
    
    def submit_task(self, task_func: Callable, *args, **kwargs) -> str:
        """Submit a task for execution.
        
        Args:
            task_func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        task_id = f"task_{int(time.time() * 1000000)}"
        
        # Get worker for task
        worker_id = self.load_balancer.get_worker_for_task()
        
        if worker_id:
            # Submit task to worker queue
            task_data = {
                'task_id': task_id,
                'func': task_func,
                'args': args,
                'kwargs': kwargs,
                'submitted_at': time.time()
            }
            
            self.load_balancer.worker_queues[worker_id].put(task_data)
            self.performance_stats['tasks_submitted'] += 1
        
        return task_id
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.performance_stats,
            'load_balancer': self.load_balancer.get_load_statistics(),
            'auto_scaler': self.auto_scaler.get_current_status()
        }


# Global scaling manager instance
scaling_manager = ScalingManager()


def with_auto_scaling(policy: Optional[ScalingPolicy] = None):
    """Decorator to enable auto-scaling for a function.
    
    Args:
        policy: Scaling policy (uses default if not provided)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not scaling_manager.auto_scaler.running:
                scaling_manager.start()
            
            task_id = scaling_manager.submit_task(func, *args, **kwargs)
            return task_id
        
        return wrapper
    
    return decorator