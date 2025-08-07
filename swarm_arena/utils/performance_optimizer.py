"""
Performance optimization utilities for large-scale sentiment-aware simulations.

Provides caching, batching, and optimization strategies for handling
1000+ concurrent sentiment-aware agents with minimal performance impact.
"""

import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import heapq

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    
    # Caching settings
    enable_sentiment_caching: bool = True
    sentiment_cache_size: int = 2000
    cache_ttl_seconds: float = 30.0
    
    # Batching settings
    enable_batch_processing: bool = True
    batch_size: int = 50
    max_batch_wait_ms: float = 5.0
    
    # Threading settings
    enable_multithreading: bool = True
    max_worker_threads: int = 4
    thread_pool_size: int = 8
    
    # Memory management
    enable_memory_management: bool = True
    max_memory_entries: int = 10000
    memory_cleanup_interval: int = 100
    
    # Spatial optimization
    enable_spatial_optimization: bool = True
    spatial_grid_size: int = 50
    max_influence_distance: float = 100.0


class LRUCache:
    """High-performance LRU cache for sentiment analysis results."""
    
    def __init__(self, capacity: int, ttl_seconds: float = 30.0):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_order = deque()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if present and not expired."""
        with self._lock:
            if key not in self.cache:
                return None
            
            value, timestamp = self.cache[key]
            
            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                self._remove(key)
                return None
            
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            
            return value
    
    def put(self, key: str, value: Any) -> None:
        """Store value in cache."""
        with self._lock:
            current_time = time.time()
            
            if key in self.cache:
                # Update existing entry
                self.cache[key] = (value, current_time)
                self.access_order.remove(key)
                self.access_order.append(key)
            else:
                # Add new entry
                if len(self.cache) >= self.capacity:
                    self._evict_lru()
                
                self.cache[key] = (value, current_time)
                self.access_order.append(key)
    
    def _remove(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self.access_order:
            lru_key = self.access_order.popleft()
            if lru_key in self.cache:
                del self.cache[lru_key]
    
    def clear_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, (_, timestamp) in self.cache.items():
                if current_time - timestamp > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove(key)
            
            return len(expired_keys)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self.cache),
                'capacity': self.capacity,
                'utilization': len(self.cache) / self.capacity,
                'ttl_seconds': self.ttl_seconds
            }


class SpatialHashGrid:
    """Spatial hash grid for efficient neighbor queries."""
    
    def __init__(self, grid_size: int, arena_width: float, arena_height: float):
        self.grid_size = grid_size
        self.arena_width = arena_width
        self.arena_height = arena_height
        
        self.cell_width = arena_width / grid_size
        self.cell_height = arena_height / grid_size
        
        self.grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.agent_positions: Dict[int, Tuple[float, float]] = {}
        
        self._lock = threading.RLock()
    
    def update_agent(self, agent_id: int, position: Tuple[float, float]) -> None:
        """Update agent position in spatial grid."""
        with self._lock:
            # Remove from old cell if exists
            if agent_id in self.agent_positions:
                old_cell = self._get_cell(self.agent_positions[agent_id])
                if agent_id in self.grid[old_cell]:
                    self.grid[old_cell].remove(agent_id)
            
            # Add to new cell
            new_cell = self._get_cell(position)
            self.grid[new_cell].append(agent_id)
            self.agent_positions[agent_id] = position
    
    def remove_agent(self, agent_id: int) -> None:
        """Remove agent from spatial grid."""
        with self._lock:
            if agent_id in self.agent_positions:
                cell = self._get_cell(self.agent_positions[agent_id])
                if agent_id in self.grid[cell]:
                    self.grid[cell].remove(agent_id)
                del self.agent_positions[agent_id]
    
    def get_neighbors(self, position: Tuple[float, float], radius: float) -> List[int]:
        """Get all agents within radius of position."""
        with self._lock:
            neighbors = []
            
            # Calculate cell range to check
            cells_to_check = self._get_cells_in_radius(position, radius)
            
            for cell in cells_to_check:
                for agent_id in self.grid[cell]:
                    agent_pos = self.agent_positions[agent_id]
                    distance = self._calculate_distance(position, agent_pos)
                    
                    if distance <= radius:
                        neighbors.append(agent_id)
            
            return neighbors
    
    def _get_cell(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Get grid cell for position."""
        x, y = position
        cell_x = min(self.grid_size - 1, max(0, int(x / self.cell_width)))
        cell_y = min(self.grid_size - 1, max(0, int(y / self.cell_height)))
        return (cell_x, cell_y)
    
    def _get_cells_in_radius(self, position: Tuple[float, float], radius: float) -> List[Tuple[int, int]]:
        """Get all cells that might contain agents within radius."""
        center_cell = self._get_cell(position)
        cells = []
        
        # Calculate cell radius
        cell_radius = int(radius / min(self.cell_width, self.cell_height)) + 1
        
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell_x = center_cell[0] + dx
                cell_y = center_cell[1] + dy
                
                if (0 <= cell_x < self.grid_size and 0 <= cell_y < self.grid_size):
                    cells.append((cell_x, cell_y))
        
        return cells
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between positions."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx * dx + dy * dy) ** 0.5
    
    def clear(self) -> None:
        """Clear all entries from grid."""
        with self._lock:
            self.grid.clear()
            self.agent_positions.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get spatial grid statistics."""
        with self._lock:
            total_agents = len(self.agent_positions)
            non_empty_cells = sum(1 for cell_agents in self.grid.values() if cell_agents)
            
            return {
                'total_agents': total_agents,
                'non_empty_cells': non_empty_cells,
                'total_cells': self.grid_size * self.grid_size,
                'cell_utilization': non_empty_cells / (self.grid_size * self.grid_size),
                'avg_agents_per_cell': total_agents / max(1, non_empty_cells)
            }


class BatchProcessor:
    """Batch processor for efficient sentiment analysis."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.pending_requests: List[Tuple[str, Any, Callable]] = []
        self.batch_timer = None
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=config.max_worker_threads)
    
    def add_request(self, key: str, data: Any, processor: Callable) -> Any:
        """Add processing request to batch."""
        if not self.config.enable_batch_processing:
            return processor(data)
        
        with self._lock:
            self.pending_requests.append((key, data, processor))
            
            # Start batch timer if not already running
            if self.batch_timer is None and len(self.pending_requests) == 1:
                self.batch_timer = threading.Timer(
                    self.config.max_batch_wait_ms / 1000.0,
                    self._process_batch
                )
                self.batch_timer.start()
            
            # Process immediately if batch is full
            if len(self.pending_requests) >= self.config.batch_size:
                if self.batch_timer:
                    self.batch_timer.cancel()
                    self.batch_timer = None
                return self._process_batch()
            
            return None  # Will be processed in batch
    
    def _process_batch(self) -> None:
        """Process accumulated batch of requests."""
        with self._lock:
            if not self.pending_requests:
                return
            
            current_batch = self.pending_requests.copy()
            self.pending_requests.clear()
            self.batch_timer = None
        
        try:
            # Group by processor type for efficiency
            processor_groups = defaultdict(list)
            for key, data, processor in current_batch:
                processor_groups[processor].append((key, data))
            
            # Process each group
            for processor, requests in processor_groups.items():
                if len(requests) == 1:
                    key, data = requests[0]
                    processor(data)
                else:
                    # Batch process if supported
                    batch_data = [data for key, data in requests]
                    try:
                        # Try batch processing
                        if hasattr(processor, 'process_batch'):
                            processor.process_batch(batch_data)
                        else:
                            # Fall back to individual processing
                            for key, data in requests:
                                processor(data)
                    except Exception as e:
                        logger.warning(f"Batch processing failed, falling back to individual: {e}")
                        for key, data in requests:
                            try:
                                processor(data)
                            except Exception as e2:
                                logger.error(f"Individual processing failed: {e2}")
        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
    
    def flush(self) -> None:
        """Force process any pending requests."""
        if self.batch_timer:
            self.batch_timer.cancel()
        self._process_batch()
    
    def shutdown(self) -> None:
        """Shutdown batch processor."""
        self.flush()
        self.executor.shutdown(wait=True)


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        
        # Initialize components
        if self.config.enable_sentiment_caching:
            self.sentiment_cache = LRUCache(
                self.config.sentiment_cache_size,
                self.config.cache_ttl_seconds
            )
        else:
            self.sentiment_cache = None
        
        if self.config.enable_batch_processing:
            self.batch_processor = BatchProcessor(self.config)
        else:
            self.batch_processor = None
        
        self.spatial_grid = None
        
        # Performance tracking
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_operations': 0,
            'individual_operations': 0,
            'spatial_queries': 0,
            'optimization_time': 0.0
        }
        
        self._cleanup_counter = 0
        
        logger.info(f"PerformanceOptimizer initialized with config: {self.config}")
    
    def initialize_spatial_grid(self, arena_width: float, arena_height: float) -> None:
        """Initialize spatial optimization grid."""
        if self.config.enable_spatial_optimization:
            self.spatial_grid = SpatialHashGrid(
                self.config.spatial_grid_size,
                arena_width,
                arena_height
            )
            logger.info(f"Initialized spatial grid: {self.config.spatial_grid_size}x{self.config.spatial_grid_size}")
    
    def get_cached_sentiment(self, cache_key: str) -> Optional[Any]:
        """Get sentiment result from cache."""
        if not self.sentiment_cache:
            return None
        
        start_time = time.time()
        result = self.sentiment_cache.get(cache_key)
        
        if result is not None:
            self.performance_stats['cache_hits'] += 1
        else:
            self.performance_stats['cache_misses'] += 1
        
        self.performance_stats['optimization_time'] += (time.time() - start_time) * 1000
        
        return result
    
    def cache_sentiment(self, cache_key: str, result: Any) -> None:
        """Store sentiment result in cache."""
        if self.sentiment_cache:
            self.sentiment_cache.put(cache_key, result)
    
    def process_sentiment_batch(self, processor: Callable, data: Any) -> Any:
        """Process sentiment analysis with batching."""
        if not self.batch_processor:
            self.performance_stats['individual_operations'] += 1
            return processor(data)
        
        cache_key = str(hash(str(data)))
        result = self.batch_processor.add_request(cache_key, data, processor)
        
        if result is None:
            self.performance_stats['batch_operations'] += 1
        else:
            self.performance_stats['individual_operations'] += 1
        
        return result
    
    def update_agent_position(self, agent_id: int, position: Tuple[float, float]) -> None:
        """Update agent position in spatial grid."""
        if self.spatial_grid:
            self.spatial_grid.update_agent(agent_id, position)
    
    def get_nearby_agents(self, position: Tuple[float, float], radius: float) -> List[int]:
        """Get nearby agents using spatial optimization."""
        if not self.spatial_grid:
            return []
        
        start_time = time.time()
        neighbors = self.spatial_grid.get_neighbors(position, radius)
        
        self.performance_stats['spatial_queries'] += 1
        self.performance_stats['optimization_time'] += (time.time() - start_time) * 1000
        
        return neighbors
    
    def periodic_cleanup(self) -> None:
        """Perform periodic cleanup operations."""
        self._cleanup_counter += 1
        
        if self._cleanup_counter % self.config.memory_cleanup_interval == 0:
            self._perform_cleanup()
    
    def _perform_cleanup(self) -> None:
        """Perform memory cleanup operations."""
        start_time = time.time()
        
        try:
            # Clear expired cache entries
            if self.sentiment_cache:
                expired_count = self.sentiment_cache.clear_expired()
                if expired_count > 0:
                    logger.debug(f"Cleaned {expired_count} expired cache entries")
            
            # Flush pending batches
            if self.batch_processor:
                self.batch_processor.flush()
            
            cleanup_time = (time.time() - start_time) * 1000
            logger.debug(f"Performance cleanup completed in {cleanup_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Performance cleanup failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add cache statistics
        if self.sentiment_cache:
            stats['cache_stats'] = self.sentiment_cache.stats()
        
        # Add spatial grid statistics
        if self.spatial_grid:
            stats['spatial_stats'] = self.spatial_grid.stats()
        
        # Calculate derived metrics
        total_cache_requests = stats['cache_hits'] + stats['cache_misses']
        if total_cache_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_requests
        else:
            stats['cache_hit_rate'] = 0.0
        
        total_operations = stats['batch_operations'] + stats['individual_operations']
        if total_operations > 0:
            stats['batch_utilization'] = stats['batch_operations'] / total_operations
        else:
            stats['batch_utilization'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_operations': 0,
            'individual_operations': 0,
            'spatial_queries': 0,
            'optimization_time': 0.0
        }
        
        logger.info("Performance statistics reset")
    
    def shutdown(self) -> None:
        """Shutdown performance optimizer."""
        if self.batch_processor:
            self.batch_processor.shutdown()
        
        logger.info("PerformanceOptimizer shutdown complete")