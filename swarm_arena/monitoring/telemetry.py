"""Real-time telemetry collection and streaming with memory management."""

import time
import threading
import numpy as np
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import json

from ..exceptions import NetworkError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TelemetryData:
    """Telemetry data point with automatic resource monitoring."""
    
    timestamp: float = field(default_factory=time.time)
    step: int = 0
    fps: float = 0.0
    active_agents: int = 0
    total_agents: int = 0
    resources_available: int = 0
    resources_collected: int = 0
    mean_reward: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "step": self.step,
            "fps": self.fps,
            "active_agents": self.active_agents,
            "total_agents": self.total_agents,
            "resources_available": self.resources_available,
            "resources_collected": self.resources_collected,
            "mean_reward": self.mean_reward,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "custom_metrics": self.custom_metrics
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    def __post_init__(self) -> None:
        """Initialize system resource monitoring."""
        if self.cpu_usage == 0.0:
            self.cpu_usage = psutil.cpu_percent(interval=None)
        if self.memory_usage == 0.0:
            self.memory_usage = psutil.virtual_memory().percent


class TelemetryCollector:
    """Collects and manages telemetry data with automatic memory management."""
    
    def __init__(self, 
                 max_history: int = 5000,  # Reduced from 10000 for memory efficiency
                 collection_interval: float = 0.1,
                 auto_start: bool = True,
                 memory_limit_mb: float = 100.0) -> None:
        """Initialize telemetry collector with memory bounds.
        
        Args:
            max_history: Maximum number of data points to keep
            collection_interval: Collection interval in seconds
            auto_start: Whether to start collection automatically
            memory_limit_mb: Memory limit for telemetry data in MB
        """
        self.max_history = max_history
        self.collection_interval = collection_interval
        self.memory_limit_mb = memory_limit_mb
        
        # Data storage with automatic memory management
        self.data_history: deque = deque(maxlen=max_history)
        self.current_data = TelemetryData()
        
        # Memory management tracking
        self._data_points_processed = 0
        self._memory_cleanup_interval = max(100, max_history // 20)
        self._last_memory_check = time.time()
        
        # Collection state
        self.collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        # Callbacks for real-time processing (with bounded list)
        self.callbacks: List[Callable[[TelemetryData], None]] = []
        self._max_callbacks = 10  # Prevent callback memory leaks
        
        # Performance tracking
        self.last_update_time = time.time()
        self.update_count = 0
        self._fps_history = deque(maxlen=60)  # 1 second of FPS history at 60 FPS
        
        if auto_start:
            self.start_collection()
        
        logger.info(f"Telemetry collector initialized (max_history={max_history}, memory_limit={memory_limit_mb}MB)")
    
    def start_collection(self) -> None:
        """Start telemetry collection thread."""
        if self.collecting:
            logger.warning("Telemetry collection already active")
            return
        
        self.collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="TelemetryCollector"
        )
        self.collection_thread.start()
        logger.info("Started telemetry collection")
    
    def stop_collection(self) -> None:
        """Stop telemetry collection thread."""
        self.collecting = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)
        logger.info("Stopped telemetry collection")
    
    def update_telemetry(self, **kwargs: Any) -> None:
        """Update current telemetry data with memory efficiency.
        
        Args:
            **kwargs: Telemetry fields to update
        """
        try:
            with self.lock:
                # Update timestamp
                self.current_data.timestamp = time.time()
                
                # Update provided fields
                for key, value in kwargs.items():
                    if hasattr(self.current_data, key):
                        setattr(self.current_data, key, value)
                    else:
                        # Limit custom metrics to prevent memory growth
                        if len(self.current_data.custom_metrics) < 20:
                            self.current_data.custom_metrics[key] = value
                        else:
                            logger.warning(f"Custom metrics limit reached, ignoring: {key}")
                
                self.update_count += 1
                self._data_points_processed += 1
                
                # Periodic memory management
                if self._data_points_processed % self._memory_cleanup_interval == 0:
                    self._perform_memory_cleanup()
                
        except Exception as e:
            logger.error(f"Failed to update telemetry: {str(e)}")
    
    def record_arena_state(self, arena) -> None:
        """Record telemetry from arena state with optimized data collection.
        
        Args:
            arena: Arena instance to collect data from
        """
        try:
            # Calculate FPS with smoothing
            current_time = time.time()
            time_delta = current_time - self.last_update_time
            fps = 1.0 / time_delta if time_delta > 0 else 0.0
            self._fps_history.append(fps)
            smoothed_fps = float(np.mean(self._fps_history)) if self._fps_history else 0.0
            self.last_update_time = current_time
            
            # Agent statistics
            active_agents = sum(1 for agent in arena.agents.values() if agent.state.alive)
            total_agents = len(arena.agents)
            
            # Environment statistics (cached for performance)
            env_stats = arena.environment.get_stats()
            resources_available = env_stats.get("active_resources", 0)
            resources_collected = env_stats.get("collected_resources", 0)
            
            # Reward statistics with memory-efficient calculation
            mean_reward = 0.0
            if arena.episode_rewards:
                recent_rewards = []
                for agent_rewards in arena.episode_rewards.values():
                    if agent_rewards:
                        # Only take last 5 rewards per agent to limit memory
                        recent_rewards.extend(agent_rewards[-5:])
                        # Limit total rewards processed
                        if len(recent_rewards) > 100:
                            break
                mean_reward = float(np.mean(recent_rewards) if recent_rewards else 0.0)
            
            # System resource monitoring (less frequent for performance)
            cpu_usage = 0.0
            memory_usage = 0.0
            if time.time() - self._last_memory_check > 1.0:  # Check every second
                try:
                    cpu_usage = psutil.cpu_percent(interval=None)
                    memory_usage = psutil.virtual_memory().percent
                    self._last_memory_check = time.time()
                except Exception:
                    pass  # Ignore psutil errors
            else:
                # Use cached values
                cpu_usage = self.current_data.cpu_usage
                memory_usage = self.current_data.memory_usage
            
            # Update telemetry
            self.update_telemetry(
                step=arena.current_step,
                fps=smoothed_fps,
                active_agents=active_agents,
                total_agents=total_agents,
                resources_available=resources_available,
                resources_collected=resources_collected,
                mean_reward=mean_reward,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage
            )
            
        except Exception as e:
            logger.error(f"Failed to record arena state: {str(e)}")
    
    def add_callback(self, callback: Callable[[TelemetryData], None]) -> None:
        """Add callback for real-time telemetry processing.
        
        Args:
            callback: Function to call with each telemetry update
        """
        if len(self.callbacks) >= self._max_callbacks:
            logger.warning(f"Maximum callbacks ({self._max_callbacks}) reached, removing oldest")
            self.callbacks.pop(0)
        
        self.callbacks.append(callback)
        logger.debug(f"Added telemetry callback: {callback.__name__}")
    
    def remove_callback(self, callback: Callable[[TelemetryData], None]) -> None:
        """Remove telemetry callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.debug(f"Removed telemetry callback: {callback.__name__}")
    
    def get_latest_data(self) -> TelemetryData:
        """Get latest telemetry data.
        
        Returns:
            Copy of current telemetry data
        """
        with self.lock:
            # Create efficient copy
            data_copy = TelemetryData(
                timestamp=self.current_data.timestamp,
                step=self.current_data.step,
                fps=self.current_data.fps,
                active_agents=self.current_data.active_agents,
                total_agents=self.current_data.total_agents,
                resources_available=self.current_data.resources_available,
                resources_collected=self.current_data.resources_collected,
                mean_reward=self.current_data.mean_reward,
                cpu_usage=self.current_data.cpu_usage,
                memory_usage=self.current_data.memory_usage,
                custom_metrics=dict(self.current_data.custom_metrics)  # Shallow copy
            )
            return data_copy
    
    def get_history(self, last_n: Optional[int] = None) -> List[TelemetryData]:
        """Get telemetry history with memory efficiency.
        
        Args:
            last_n: Number of most recent entries (None for all, max 1000)
            
        Returns:
            List of telemetry data points
        """
        with self.lock:
            history = list(self.data_history)
            if last_n is not None:
                # Limit maximum return size for memory efficiency
                last_n = min(last_n, 1000)
                history = history[-last_n:]
            else:
                # Return at most 1000 points if no limit specified
                history = history[-1000:]
            return history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get telemetry statistics with performance metrics.
        
        Returns:
            Dictionary of statistics
        """
        with self.lock:
            if not self.data_history:
                return {"status": "no_data"}
            
            # Calculate statistics from recent history for performance
            recent_data = list(self.data_history)[-100:]  # Last 100 points
            fps_values = [d.fps for d in recent_data if d.fps > 0]
            reward_values = [d.mean_reward for d in recent_data]
            
            stats = {
                "data_points": len(self.data_history),
                "max_history": self.max_history,
                "collection_time": time.time() - (self.data_history[0].timestamp if self.data_history else time.time()),
                "avg_fps": float(np.mean(fps_values)) if fps_values else 0.0,
                "min_fps": float(np.min(fps_values)) if fps_values else 0.0,
                "max_fps": float(np.max(fps_values)) if fps_values else 0.0,
                "avg_reward": float(np.mean(reward_values)) if reward_values else 0.0,
                "update_count": self.update_count,
                "callbacks_count": len(self.callbacks),
                "memory_cleanups": self._data_points_processed // self._memory_cleanup_interval,
                "estimated_memory_mb": len(self.data_history) * 0.001  # Rough estimate
            }
            
            return stats
    
    def export_data(self, format: str = "json", compress: bool = True) -> str:
        """Export telemetry data with compression option.
        
        Args:
            format: Export format ('json', 'csv')
            compress: Whether to compress large datasets
            
        Returns:
            Exported data as string
        """
        with self.lock:
            # Limit export size for memory efficiency
            export_data = list(self.data_history)[-1000:]  # Last 1000 points max
            
            if format.lower() == "json":
                data_list = [data.to_dict() for data in export_data]
                return json.dumps(data_list, indent=2 if not compress else None)
            
            elif format.lower() == "csv":
                if not export_data:
                    return "No data available"
                
                # CSV header
                first_data = export_data[0]
                headers = list(first_data.to_dict().keys())
                csv_lines = [','.join(headers)]
                
                # CSV data
                for data in export_data:
                    data_dict = data.to_dict()
                    row = [str(data_dict.get(header, '')) for header in headers]
                    csv_lines.append(','.join(row))
                
                return '\n'.join(csv_lines)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def _perform_memory_cleanup(self) -> None:
        """Perform memory cleanup operations."""
        try:
            # Clear custom metrics if they're growing too large
            metrics_size = sum(len(str(v)) for v in self.current_data.custom_metrics.values())
            if metrics_size > 1000:  # 1KB limit
                old_size = len(self.current_data.custom_metrics)
                # Keep only the most recent metrics
                self.current_data.custom_metrics = dict(
                    list(self.current_data.custom_metrics.items())[-10:]
                )
                logger.debug(f"Cleaned custom metrics: {old_size} -> {len(self.current_data.custom_metrics)}")
            
            # Force garbage collection of FPS history if needed
            if len(self._fps_history) > 100:
                self._fps_history = deque(list(self._fps_history)[-60:], maxlen=60)
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {str(e)}")
    
    def _collection_loop(self) -> None:
        """Background collection loop with error recovery."""
        logger.debug("Telemetry collection loop started")
        error_count = 0
        max_errors = 10
        
        while self.collecting and error_count < max_errors:
            try:
                # Store current data in history
                with self.lock:
                    data_copy = self.get_latest_data()
                    self.data_history.append(data_copy)
                
                # Call registered callbacks with error isolation
                for callback in self.callbacks[:]:  # Copy list to avoid modification during iteration
                    try:
                        callback(data_copy)
                    except Exception as e:
                        logger.warning(f"Telemetry callback {callback.__name__} failed: {str(e)}")
                        # Remove failing callbacks after 3 consecutive failures
                        if not hasattr(callback, '_error_count'):
                            callback._error_count = 0
                        callback._error_count += 1
                        if callback._error_count >= 3:
                            logger.warning(f"Removing failing callback: {callback.__name__}")
                            self.callbacks.remove(callback)
                
                time.sleep(self.collection_interval)
                error_count = 0  # Reset error count on successful iteration
                
            except Exception as e:
                error_count += 1
                logger.error(f"Telemetry collection loop error ({error_count}/{max_errors}): {str(e)}")
                time.sleep(min(self.collection_interval * error_count, 5.0))  # Exponential backoff
        
        if error_count >= max_errors:
            logger.error("Too many telemetry collection errors, stopping collection")
            self.collecting = False
        
        logger.debug("Telemetry collection loop stopped")
    
    def cleanup(self) -> None:
        """Explicit cleanup of resources."""
        try:
            self.stop_collection()
            self.data_history.clear()
            self.callbacks.clear()
            self.current_data.custom_metrics.clear()
            self._fps_history.clear()
            logger.info("Telemetry collector cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass