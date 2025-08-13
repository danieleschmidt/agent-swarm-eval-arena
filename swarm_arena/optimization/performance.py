"""Performance optimization utilities and algorithms."""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import numpy as np
from dataclasses import dataclass
from functools import lru_cache, wraps
import gc
import psutil
import queue


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    
    # Threading/Processing
    max_workers: int = mp.cpu_count()
    use_processes: bool = False
    chunk_size: int = 100
    
    # Caching
    cache_size: int = 1000
    cache_ttl: float = 300.0  # 5 minutes
    
    # Memory management
    gc_threshold: float = 0.8  # Trigger GC at 80% memory usage
    max_memory_gb: float = 4.0
    
    # Vectorization
    use_vectorization: bool = True
    batch_size: int = 1000


class MemoryManager:
    """Manages memory usage and triggers cleanup when needed."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.peak_memory = 0.0
        self.gc_count = 0
        
    def check_memory_usage(self) -> float:
        """Check current memory usage as fraction of total."""
        memory = psutil.virtual_memory()
        current_gb = memory.used / (1024**3)
        
        self.peak_memory = max(self.peak_memory, current_gb)
        
        return memory.percent / 100.0
    
    def should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        usage = self.check_memory_usage()
        return usage > self.config.gc_threshold
    
    def cleanup_memory(self) -> int:
        """Force garbage collection and return objects collected."""
        self.gc_count += 1
        return gc.collect()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        memory = psutil.virtual_memory()
        
        return {
            'current_gb': memory.used / (1024**3),
            'peak_gb': self.peak_memory,
            'usage_percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'gc_count': self.gc_count
        }


class AdaptiveCache:
    """Cache with adaptive TTL and automatic cleanup."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_counts: Dict[str, int] = {}
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if time.time() > entry['expires_at']:
                del self.cache[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                self.misses += 1
                return None
            
            # Update access count and hit stats
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.hits += 1
            
            # Adaptive TTL: extend TTL for frequently accessed items
            if self.access_counts[key] > 10:
                entry['expires_at'] = time.time() + (self.default_ttl * 1.5)
            
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self.lock:
            # Clean up expired entries first
            self._cleanup_expired()
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            ttl = ttl or self.default_ttl
            self.cache[key] = {
                'value': value,
                'created_at': time.time(),
                'expires_at': time.time() + ttl
            }
            
            # Reset access count
            self.access_counts[key] = 1
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time > entry['expires_at']
        ]
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_counts:
                del self.access_counts[key]
        
        return len(expired_keys)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.cache:
            return
        
        # Find item with lowest access count
        lru_key = min(self.access_counts.items(), key=lambda x: x[1])[0]
        
        del self.cache[lru_key]
        del self.access_counts[lru_key]
        self.evictions += 1
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_counts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'memory_usage': sum(
                    len(str(entry)) for entry in self.cache.values()
                )
            }


class VectorizedOperations:
    """Optimized vectorized operations for agent simulations."""
    
    @staticmethod
    def distance_matrix(positions: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between all positions.
        
        Args:
            positions: Array of shape (n_agents, 2) with agent positions
            
        Returns:
            Distance matrix of shape (n_agents, n_agents)
        """
        # Vectorized distance computation
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        return distances
    
    @staticmethod
    def find_neighbors(positions: np.ndarray, radius: float) -> List[List[int]]:
        """Find neighbors within radius for each agent.
        
        Args:
            positions: Array of shape (n_agents, 2)
            radius: Search radius
            
        Returns:
            List of neighbor indices for each agent
        """
        distances = VectorizedOperations.distance_matrix(positions)
        
        neighbors = []
        for i in range(len(positions)):
            # Find agents within radius (excluding self)
            neighbor_mask = (distances[i] <= radius) & (distances[i] > 0)
            neighbor_indices = np.where(neighbor_mask)[0].tolist()
            neighbors.append(neighbor_indices)
        
        return neighbors
    
    @staticmethod
    def batch_update_positions(positions: np.ndarray, 
                             velocities: np.ndarray,
                             dt: float,
                             boundaries: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """Update all agent positions in a single vectorized operation.
        
        Args:
            positions: Current positions (n_agents, 2)
            velocities: Current velocities (n_agents, 2)
            dt: Time step
            boundaries: (min_x, max_x, min_y, max_y) for boundary checking
            
        Returns:
            Updated positions
        """
        new_positions = positions + velocities * dt
        
        if boundaries:
            min_x, max_x, min_y, max_y = boundaries
            new_positions[:, 0] = np.clip(new_positions[:, 0], min_x, max_x)
            new_positions[:, 1] = np.clip(new_positions[:, 1], min_y, max_y)
        
        return new_positions
    
    @staticmethod
    def compute_forces(positions: np.ndarray,
                      force_func: Callable[[np.ndarray], np.ndarray],
                      batch_size: int = 1000) -> np.ndarray:
        """Compute forces for all agents in batches.
        
        Args:
            positions: Agent positions
            force_func: Function to compute forces
            batch_size: Size of processing batches
            
        Returns:
            Computed forces for all agents
        """
        n_agents = len(positions)
        forces = np.zeros_like(positions)
        
        for i in range(0, n_agents, batch_size):
            end_idx = min(i + batch_size, n_agents)
            batch_positions = positions[i:end_idx]
            batch_forces = force_func(batch_positions)
            forces[i:end_idx] = batch_forces
        
        return forces


class ParallelExecutor:
    """Parallel execution manager for CPU-intensive tasks."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
    def __enter__(self):
        if self.config.use_processes:
            self.process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers)
            return self.process_pool
        else:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
            return self.thread_pool
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
    
    def map_parallel(self, func: Callable, items: List[Any]) -> List[Any]:
        """Execute function on items in parallel.
        
        Args:
            func: Function to execute
            items: Items to process
            
        Returns:
            Results from parallel execution
        """
        with self as executor:
            # Split into chunks for better load balancing
            chunk_size = max(1, len(items) // self.config.max_workers)
            chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
            
            # Submit chunk processing tasks
            futures = [executor.submit(self._process_chunk, func, chunk) for chunk in chunks]
            
            # Collect results
            results = []
            for future in futures:
                results.extend(future.result())
            
            return results
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        return [func(item) for item in chunk]


class ResourcePool:
    """Pool of reusable resources to avoid allocation overhead."""
    
    def __init__(self, factory: Callable[[], Any], max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self.pool: queue.Queue = queue.Queue(maxsize=max_size)
        self.created_count = 0
        self.lock = threading.Lock()
    
    def acquire(self) -> Any:
        """Acquire a resource from the pool."""
        try:
            return self.pool.get_nowait()
        except queue.Empty:
            with self.lock:
                if self.created_count < self.max_size:
                    self.created_count += 1
                    return self.factory()
                else:
                    # Pool is full and empty, wait for a resource
                    return self.pool.get()
    
    def release(self, resource: Any) -> None:
        """Release a resource back to the pool."""
        try:
            self.pool.put_nowait(resource)
        except queue.Full:
            # Pool is full, discard the resource
            pass
    
    def size(self) -> int:
        """Get current pool size."""
        return self.pool.qsize()


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.memory_manager = MemoryManager(self.config)
        self.cache = AdaptiveCache(self.config.cache_size, self.config.cache_ttl)
        self.vectorops = VectorizedOperations()
        
        # Resource pools
        self.array_pool = ResourcePool(
            lambda: np.empty((self.config.batch_size, 2), dtype=np.float32)
        )
        
        # Performance tracking
        self.optimization_stats = {
            'cache_optimizations': 0,
            'vectorization_optimizations': 0,
            'parallel_optimizations': 0,
            'memory_optimizations': 0
        }
    
    def optimize_computation(self, func: Callable) -> Callable:
        """Decorator to optimize computation with caching and memory management.
        
        Args:
            func: Function to optimize
            
        Returns:
            Optimized function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try cache first
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.optimization_stats['cache_optimizations'] += 1
                return cached_result
            
            # Check memory before computation
            if self.memory_manager.should_trigger_gc():
                self.memory_manager.cleanup_memory()
                self.optimization_stats['memory_optimizations'] += 1
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            self.cache.set(cache_key, result)
            
            return result
        
        return wrapper
    
    def optimize_array_operations(self, positions: np.ndarray, 
                                 operation: str, 
                                 **kwargs) -> Any:
        """Optimize array operations with vectorization.
        
        Args:
            positions: Position array
            operation: Operation type
            **kwargs: Additional parameters
            
        Returns:
            Optimized operation result
        """
        self.optimization_stats['vectorization_optimizations'] += 1
        
        if operation == "distances":
            return self.vectorops.distance_matrix(positions)
        
        elif operation == "neighbors":
            radius = kwargs.get('radius', 50.0)
            return self.vectorops.find_neighbors(positions, radius)
        
        elif operation == "update_positions":
            velocities = kwargs['velocities']
            dt = kwargs.get('dt', 1.0)
            boundaries = kwargs.get('boundaries')
            return self.vectorops.batch_update_positions(positions, velocities, dt, boundaries)
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def execute_parallel(self, func: Callable, items: List[Any]) -> List[Any]:
        """Execute function in parallel.
        
        Args:
            func: Function to execute
            items: Items to process
            
        Returns:
            Results from parallel execution
        """
        self.optimization_stats['parallel_optimizations'] += 1
        
        executor = ParallelExecutor(self.config)
        return executor.map_parallel(func, items)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics.
        
        Returns:
            Dictionary with optimization stats
        """
        return {
            **self.optimization_stats,
            'cache_stats': self.cache.get_stats(),
            'memory_stats': self.memory_manager.get_memory_stats(),
            'config': {
                'max_workers': self.config.max_workers,
                'use_processes': self.config.use_processes,
                'cache_size': self.config.cache_size,
                'vectorization_enabled': self.config.use_vectorization
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.cache.clear()
        self.memory_manager.cleanup_memory()


# Global optimizer instance
global_optimizer = PerformanceOptimizer()


# Convenience decorators

def cached(ttl: float = 300.0):
    """Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            result = global_optimizer.cache.get(cache_key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            global_optimizer.cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator


def vectorized(func):
    """Decorator to indicate function uses vectorized operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        global_optimizer.optimization_stats['vectorization_optimizations'] += 1
        return func(*args, **kwargs)
    
    return wrapper


def memory_managed(func):
    """Decorator to add memory management to function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if global_optimizer.memory_manager.should_trigger_gc():
            global_optimizer.memory_manager.cleanup_memory()
            global_optimizer.optimization_stats['memory_optimizations'] += 1
        
        return func(*args, **kwargs)
    
    return wrapper


class BatchProcessor:
    """Processes items in optimized batches."""
    
    def __init__(self, batch_size: int = 1000, parallel: bool = True):
        self.batch_size = batch_size
        self.parallel = parallel
        
    def process(self, items: List[Any], processor: Callable[[List[Any]], List[Any]]) -> List[Any]:
        """Process items in batches.
        
        Args:
            items: Items to process
            processor: Function that processes a batch
            
        Returns:
            Processed results
        """
        batches = [
            items[i:i + self.batch_size] 
            for i in range(0, len(items), self.batch_size)
        ]
        
        if self.parallel and len(batches) > 1:
            # Process batches in parallel
            results = global_optimizer.execute_parallel(processor, batches)
            # Flatten results
            flattened = []
            for batch_result in results:
                flattened.extend(batch_result)
            return flattened
        else:
            # Process batches sequentially
            results = []
            for batch in batches:
                results.extend(processor(batch))
            return results