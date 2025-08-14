"""Advanced performance optimization engine for Swarm Arena."""

import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import lru_cache, wraps

from ..utils.logging import get_logger
from ..resilience.circuit_breaker import circuit_breaker, CONFIGS

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    steps_per_second: float = 0.0
    agent_throughput: float = 0.0
    latency_p95: float = 0.0
    cache_hit_rate: float = 0.0
    gc_pressure: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    enable_adaptive_batching: bool = True
    enable_memory_pooling: bool = True
    enable_vectorization: bool = True
    enable_caching: bool = True
    enable_parallel_processing: bool = True
    max_worker_threads: int = 8
    batch_size_min: int = 32
    batch_size_max: int = 512
    cache_size_limit: int = 10000
    memory_threshold: float = 0.8  # 80% memory usage threshold
    cpu_threshold: float = 0.9     # 90% CPU usage threshold


class AdaptiveBatchProcessor:
    """Dynamically adjusts batch sizes based on performance."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.current_batch_size = config.batch_size_min * 2
        self.performance_history = deque(maxlen=100)
        self.last_adjustment = time.time()
        self.adjustment_cooldown = 5.0  # seconds
    
    def process_batch(self, items: List[Any], processor_func: Callable) -> List[Any]:
        """Process items in adaptive batches."""
        if not items:
            return []
        
        results = []
        start_time = time.time()
        
        # Process in current batch size
        for i in range(0, len(items), self.current_batch_size):
            batch = items[i:i + self.current_batch_size]
            batch_start = time.time()
            
            try:
                batch_results = processor_func(batch)
                results.extend(batch_results)
                
                # Record batch performance
                batch_duration = time.time() - batch_start
                throughput = len(batch) / batch_duration if batch_duration > 0 else 0
                
                self.performance_history.append({
                    'batch_size': len(batch),
                    'duration': batch_duration,
                    'throughput': throughput,
                    'timestamp': batch_start
                })
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Process items individually as fallback
                for item in batch:
                    try:
                        results.append(processor_func([item])[0])
                    except Exception:
                        pass  # Skip failed items
        
        total_duration = time.time() - start_time
        
        # Adapt batch size based on performance
        self._adapt_batch_size(total_duration, len(items))
        
        return results
    
    def _adapt_batch_size(self, total_duration: float, total_items: int) -> None:
        """Adapt batch size based on recent performance."""
        now = time.time()
        
        # Only adjust if cooldown period has passed
        if now - self.last_adjustment < self.adjustment_cooldown:
            return
        
        if len(self.performance_history) < 10:
            return
        
        # Calculate recent performance metrics
        recent_metrics = list(self.performance_history)[-10:]
        avg_throughput = np.mean([m['throughput'] for m in recent_metrics])
        throughput_std = np.std([m['throughput'] for m in recent_metrics])
        
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent / 100.0
        
        # Adaptation logic
        if cpu_usage > self.config.cpu_threshold:
            # High CPU usage - reduce batch size
            self.current_batch_size = max(
                self.config.batch_size_min,
                int(self.current_batch_size * 0.8)
            )
            logger.debug(f"Reduced batch size to {self.current_batch_size} due to high CPU usage")
            
        elif memory_usage > self.config.memory_threshold:
            # High memory usage - reduce batch size
            self.current_batch_size = max(
                self.config.batch_size_min,
                int(self.current_batch_size * 0.7)
            )
            logger.debug(f"Reduced batch size to {self.current_batch_size} due to high memory usage")
            
        elif throughput_std / avg_throughput < 0.1 and avg_throughput > 0:
            # Stable performance - try increasing batch size
            self.current_batch_size = min(
                self.config.batch_size_max,
                int(self.current_batch_size * 1.2)
            )
            logger.debug(f"Increased batch size to {self.current_batch_size} due to stable performance")
        
        self.last_adjustment = now


class MemoryPool:
    """Memory pool for efficient object reuse."""
    
    def __init__(self, obj_type: type, initial_size: int = 100, max_size: int = 1000):
        self.obj_type = obj_type
        self.max_size = max_size
        self.pool = []
        self.in_use = set()
        self.lock = threading.Lock()
        
        # Pre-populate pool
        for _ in range(initial_size):
            self.pool.append(self._create_object())
    
    def acquire(self) -> Any:
        """Acquire an object from the pool."""
        with self.lock:
            if self.pool:
                obj = self.pool.pop()
            else:
                obj = self._create_object()
            
            self.in_use.add(id(obj))
            return obj
    
    def release(self, obj: Any) -> None:
        """Release an object back to the pool."""
        with self.lock:
            obj_id = id(obj)
            if obj_id in self.in_use:
                self.in_use.remove(obj_id)
                
                if len(self.pool) < self.max_size:
                    self._reset_object(obj)
                    self.pool.append(obj)
    
    def _create_object(self) -> Any:
        """Create a new object instance."""
        if self.obj_type == np.ndarray:
            return np.zeros((2,), dtype=np.float32)
        else:
            return self.obj_type()
    
    def _reset_object(self, obj: Any) -> None:
        """Reset object to initial state."""
        if isinstance(obj, np.ndarray):
            obj.fill(0)
        elif hasattr(obj, 'reset'):
            obj.reset()
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self.lock:
            return {
                'available': len(self.pool),
                'in_use': len(self.in_use),
                'total_created': len(self.pool) + len(self.in_use)
            }


class VectorizedOperations:
    """Vectorized operations for high-performance computation."""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def compute_distances_vectorized(positions: tuple, targets: tuple) -> np.ndarray:
        """Compute distances between positions and targets using vectorization."""
        pos_array = np.array(positions)
        target_array = np.array(targets)
        
        if pos_array.ndim == 1:
            pos_array = pos_array.reshape(1, -1)
        if target_array.ndim == 1:
            target_array = target_array.reshape(1, -1)
        
        # Vectorized distance computation
        diff = pos_array[:, np.newaxis, :] - target_array[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        return distances
    
    @staticmethod
    def batch_agent_updates(positions: np.ndarray, velocities: np.ndarray, 
                          actions: np.ndarray, dt: float = 1/60) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized agent position and velocity updates."""
        # Action to movement mapping (vectorized)
        action_vectors = np.array([
            [0, 0],    # no-op
            [0, 1],    # move_up  
            [0, -1],   # move_down
            [-1, 0],   # move_left
            [1, 0],    # move_right
            [0, 0],    # collect_resource
        ])
        
        # Apply actions to velocities
        new_velocities = action_vectors[actions]
        
        # Update positions
        new_positions = positions + new_velocities * dt
        
        return new_positions, new_velocities
    
    @staticmethod
    def spatial_hash_vectorized(positions: np.ndarray, cell_size: float) -> Dict[int, List[int]]:
        """Vectorized spatial hashing for neighbor queries."""
        # Convert positions to grid coordinates
        grid_coords = (positions / cell_size).astype(np.int32)
        
        # Compute hash values
        hash_values = grid_coords[:, 0] * 73856093 + grid_coords[:, 1] * 19349663
        
        # Group agents by hash
        spatial_hash = defaultdict(list)
        for i, hash_val in enumerate(hash_values):
            spatial_hash[hash_val].append(i)
        
        return dict(spatial_hash)


class ParallelExecutor:
    """Parallel execution engine for concurrent processing."""
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_futures = []
    
    def submit_parallel_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit a task for parallel execution."""
        future = self.executor.submit(func, *args, **kwargs)
        self.active_futures.append(future)
        return future
    
    def map_parallel(self, func: Callable, items: List[Any], chunk_size: Optional[int] = None) -> List[Any]:
        """Execute function over items in parallel."""
        if chunk_size is None:
            chunk_size = max(1, len(items) // self.max_workers)
        
        # Split items into chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Submit chunks for processing
        futures = []
        for chunk in chunks:
            future = self.executor.submit(lambda c: [func(item) for item in c], chunk)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                chunk_results = future.result(timeout=30)
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"Parallel execution failed: {e}")
        
        return results
    
    def wait_for_completion(self, timeout: float = 60.0) -> None:
        """Wait for all active tasks to complete."""
        completed_futures = []
        for future in self.active_futures:
            try:
                future.result(timeout=timeout)
                completed_futures.append(future)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                completed_futures.append(future)
        
        # Remove completed futures
        for future in completed_futures:
            self.active_futures.remove(future)
    
    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class PerformanceOptimizer:
    """Main performance optimization engine."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self.batch_processor = AdaptiveBatchProcessor(config)
        self.memory_pools = {}
        self.parallel_executor = ParallelExecutor(config.max_worker_threads)
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Performance monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        
        logger.info(f"Performance optimizer initialized with config: {config}")
    
    def start_monitoring(self) -> None:
        """Start performance monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_performance(self) -> None:
        """Monitor system performance continuously."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self.metrics.cpu_usage = psutil.cpu_percent(interval=1.0) / 100.0
                self.metrics.memory_usage = psutil.virtual_memory().percent / 100.0
                
                # GPU metrics (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.metrics.gpu_usage = max(gpu.load for gpu in gpus)
                except ImportError:
                    pass
                
                # Cache hit rate
                total_cache_ops = self.cache_stats['hits'] + self.cache_stats['misses']
                if total_cache_ops > 0:
                    self.metrics.cache_hit_rate = self.cache_stats['hits'] / total_cache_ops
                
                # GC pressure
                gc_stats = gc.get_stats()
                if gc_stats:
                    self.metrics.gc_pressure = sum(stat['collections'] for stat in gc_stats)
                
                # Auto-optimization based on metrics
                self._auto_optimize()
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
            
            time.sleep(5.0)  # Monitor every 5 seconds
    
    def _auto_optimize(self) -> None:
        """Automatically optimize based on current metrics."""
        # Trigger garbage collection if high memory pressure
        if self.metrics.memory_usage > 0.85:
            gc.collect()
            logger.debug("Triggered garbage collection due to high memory usage")
        
        # Adjust cache sizes if memory pressure is high
        if self.metrics.memory_usage > 0.80:
            self._reduce_cache_sizes()
        
        # Log performance warnings
        if self.metrics.cpu_usage > 0.95:
            logger.warning(f"High CPU usage: {self.metrics.cpu_usage:.1%}")
        
        if self.metrics.memory_usage > 0.90:
            logger.warning(f"High memory usage: {self.metrics.memory_usage:.1%}")
    
    def _reduce_cache_sizes(self) -> None:
        """Reduce cache sizes to free memory."""
        # Clear LRU caches
        VectorizedOperations.compute_distances_vectorized.cache_clear()
        logger.debug("Cleared function caches due to memory pressure")
    
    def get_memory_pool(self, obj_type: type, initial_size: int = 100) -> MemoryPool:
        """Get or create memory pool for object type."""
        pool_key = obj_type.__name__
        if pool_key not in self.memory_pools:
            self.memory_pools[pool_key] = MemoryPool(obj_type, initial_size)
        return self.memory_pools[pool_key]
    
    def process_agents_optimized(self, agents: List[Any], processor_func: Callable) -> List[Any]:
        """Process agents with all optimizations enabled."""
        if not self.config.enable_adaptive_batching:
            return [processor_func(agent) for agent in agents]
        
        # Use adaptive batching
        return self.batch_processor.process_batch(agents, processor_func)
    
    @circuit_breaker("performance_optimizer", CONFIGS["simulation"])
    def optimize_simulation_step(self, step_func: Callable, *args, **kwargs) -> Any:
        """Optimize a simulation step with circuit breaker protection."""
        start_time = time.time()
        
        try:
            # Pre-optimization
            if self.config.enable_memory_pooling:
                # Use memory pools for temporary objects
                pass
            
            # Execute step
            result = step_func(*args, **kwargs)
            
            # Post-optimization
            duration = time.time() - start_time
            if duration > 0:
                self.metrics.steps_per_second = 1.0 / duration
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized simulation step failed: {e}")
            raise
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        memory_pool_stats = {name: pool.get_stats() 
                           for name, pool in self.memory_pools.items()}
        
        return {
            "metrics": {
                "cpu_usage": f"{self.metrics.cpu_usage:.1%}",
                "memory_usage": f"{self.metrics.memory_usage:.1%}",
                "gpu_usage": f"{self.metrics.gpu_usage:.1%}",
                "steps_per_second": f"{self.metrics.steps_per_second:.1f}",
                "cache_hit_rate": f"{self.metrics.cache_hit_rate:.1%}",
                "gc_pressure": self.metrics.gc_pressure,
            },
            "memory_pools": memory_pool_stats,
            "batch_processor": {
                "current_batch_size": self.batch_processor.current_batch_size,
                "performance_samples": len(self.batch_processor.performance_history),
            },
            "parallel_executor": {
                "max_workers": self.parallel_executor.max_workers,
                "active_futures": len(self.parallel_executor.active_futures),
            },
            "cache_stats": self.cache_stats,
            "optimizations_enabled": {
                "adaptive_batching": self.config.enable_adaptive_batching,
                "memory_pooling": self.config.enable_memory_pooling,
                "vectorization": self.config.enable_vectorization,
                "caching": self.config.enable_caching,
                "parallel_processing": self.config.enable_parallel_processing,
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown the performance optimizer."""
        self.stop_monitoring()
        self.parallel_executor.shutdown()
        logger.info("Performance optimizer shutdown complete")


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer(OptimizationConfig())