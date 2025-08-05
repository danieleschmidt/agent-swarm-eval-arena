"""Performance optimization utilities and caching systems."""

import time
import functools
import threading
from typing import Any, Dict, Optional, Callable, TypeVar, Generic
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class CacheStats:
    """Cache performance statistics."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class LRUCache:
    """Thread-safe LRU cache with performance monitoring."""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl: Time-to-live in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[Any, float] = {}
        self.stats = CacheStats()
        self.lock = threading.RLock()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            self.stats.total_requests += 1
            
            # Check if key exists
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            # Check TTL expiration
            if self.ttl is not None:
                age = time.time() - self.timestamps[key]
                if age > self.ttl:
                    # Remove expired item
                    del self.cache[key]
                    del self.timestamps[key]
                    self.stats.misses += 1
                    return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.stats.hits += 1
            return value
    
    def put(self, key: Any, value: Any) -> None:
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
            
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = time.time()
            
            # Evict oldest if necessary
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
                self.stats.evictions += 1
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


def cached(max_size: int = 128, ttl: Optional[float] = None, 
          key_func: Optional[Callable] = None) -> Callable[[F], F]:
    """Decorator for caching function results.
    
    Args:
        max_size: Maximum cache size
        ttl: Time-to-live in seconds
        key_func: Custom key generation function
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: F) -> F:
        cache = LRUCache(max_size=max_size, ttl=ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = (args, tuple(sorted(kwargs.items())))
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        # Add cache management methods
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.get_stats
        wrapper.cache_size = cache.size
        
        return wrapper
    return decorator


class PerformanceProfiler:
    """Performance profiling and optimization utilities."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.timings: Dict[str, list] = {}
        self.counters: Dict[str, int] = {}
        self.lock = threading.RLock()
    
    def time_function(self, name: str):
        """Context manager for timing function execution.
        
        Args:
            name: Name for the timing measurement
        """
        return self._TimingContext(self, name)
    
    def record_timing(self, name: str, duration: float) -> None:
        """Record a timing measurement.
        
        Args:
            name: Name of the measurement
            duration: Duration in seconds
        """
        with self.lock:
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)
            
            # Keep only recent measurements to prevent memory growth
            if len(self.timings[name]) > 1000:
                self.timings[name] = self.timings[name][-500:]
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a performance counter.
        
        Args:
            name: Counter name
            value: Increment value
        """
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + value
    
    def get_timing_stats(self, name: str) -> Dict[str, float]:
        """Get timing statistics for a measurement.
        
        Args:
            name: Measurement name
            
        Returns:
            Dictionary of timing statistics
        """
        with self.lock:
            if name not in self.timings or not self.timings[name]:
                return {"count": 0}
            
            measurements = self.timings[name]
            return {
                "count": len(measurements),
                "total": sum(measurements),
                "mean": np.mean(measurements),
                "median": np.median(measurements),
                "min": min(measurements),
                "max": max(measurements),
                "std": np.std(measurements),
                "p95": np.percentile(measurements, 95),
                "p99": np.percentile(measurements, 99)
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all performance statistics.
        
        Returns:
            Complete performance statistics
        """
        with self.lock:
            stats = {
                "timings": {},
                "counters": dict(self.counters)
            }
            
            for name in self.timings:
                stats["timings"][name] = self.get_timing_stats(name)
            
            return stats
    
    def reset(self) -> None:
        """Reset all performance measurements."""
        with self.lock:
            self.timings.clear()
            self.counters.clear()
    
    class _TimingContext:
        """Context manager for timing measurements."""
        
        def __init__(self, profiler: 'PerformanceProfiler', name: str):
            self.profiler = profiler
            self.name = name
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time is not None:
                duration = time.time() - self.start_time
                self.profiler.record_timing(self.name, duration)


def profile_function(name: Optional[str] = None):
    """Decorator for automatic function profiling.
    
    Args:
        name: Custom name for profiling (defaults to function name)
    """
    def decorator(func: F) -> F:
        profiler_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with global_profiler.time_function(profiler_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class BatchProcessor:
    """Batch processing utilities for improved performance."""
    
    def __init__(self, batch_size: int = 100, max_wait_time: float = 0.1):
        """Initialize batch processor.
        
        Args:
            batch_size: Maximum items per batch
            max_wait_time: Maximum wait time before processing partial batch
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_items = []
        self.last_process_time = time.time()
        self.lock = threading.RLock()
    
    def add_item(self, item: Any) -> None:
        """Add item to batch.
        
        Args:
            item: Item to add to batch
        """
        with self.lock:
            self.pending_items.append(item)
    
    def should_process_batch(self) -> bool:
        """Check if batch should be processed.
        
        Returns:
            True if batch should be processed
        """
        with self.lock:
            if len(self.pending_items) >= self.batch_size:
                return True
            
            if (self.pending_items and 
                time.time() - self.last_process_time >= self.max_wait_time):
                return True
            
            return False
    
    def get_batch(self) -> list:
        """Get current batch and clear pending items.
        
        Returns:
            List of batched items
        """
        with self.lock:
            batch = self.pending_items.copy()
            self.pending_items.clear()
            self.last_process_time = time.time()
            return batch


class MemoryPool:
    """Memory pool for object reuse and reduced allocation overhead."""
    
    def __init__(self, object_factory: Callable, initial_size: int = 10, max_size: int = 100):
        """Initialize memory pool.
        
        Args:
            object_factory: Function to create new objects
            initial_size: Initial pool size
            max_size: Maximum pool size
        """
        self.object_factory = object_factory
        self.max_size = max_size
        self.pool = []
        self.lock = threading.RLock()
        
        # Pre-populate pool
        for _ in range(initial_size):
            self.pool.append(object_factory())
    
    def get_object(self):
        """Get object from pool or create new one.
        
        Returns:
            Reusable object
        """
        with self.lock:
            if self.pool:
                return self.pool.pop()
            else:
                return self.object_factory()
    
    def return_object(self, obj) -> None:
        """Return object to pool.
        
        Args:
            obj: Object to return to pool
        """
        with self.lock:
            if len(self.pool) < self.max_size:
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)
    
    def size(self) -> int:
        """Get current pool size."""
        return len(self.pool)


# Global performance profiler instance
global_profiler = PerformanceProfiler()


# Optimized distance calculation functions
@cached(max_size=10000, ttl=1.0)
def cached_distance(pos1: tuple, pos2: tuple) -> float:
    """Cached distance calculation.
    
    Args:
        pos1: First position as tuple
        pos2: Second position as tuple
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def vectorized_distances(positions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Vectorized distance calculation for multiple positions.
    
    Args:
        positions: Array of positions (N, 2)
        target: Target position (2,)
        
    Returns:
        Array of distances (N,)
    """
    diff = positions - target
    return np.sqrt(np.sum(diff**2, axis=1))


def optimized_neighbor_search(positions: Dict[int, np.ndarray], 
                            query_pos: np.ndarray, 
                            radius: float) -> list:
    """Optimized neighbor search using vectorized operations.
    
    Args:
        positions: Dictionary of agent positions
        query_pos: Query position
        radius: Search radius
        
    Returns:
        List of agent IDs within radius
    """
    if not positions:
        return []
    
    # Convert to arrays for vectorized operations
    agent_ids = list(positions.keys())
    pos_array = np.array([positions[aid] for aid in agent_ids])
    
    # Vectorized distance calculation
    distances = vectorized_distances(pos_array, query_pos)
    
    # Find neighbors within radius
    neighbor_mask = distances <= radius
    neighbor_indices = np.where(neighbor_mask)[0]
    
    return [agent_ids[i] for i in neighbor_indices]