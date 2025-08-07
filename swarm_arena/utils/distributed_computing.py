"""
Distributed computing utilities for large-scale sentiment-aware multi-agent simulations.

Provides Ray-based distributed processing, worker management, and fault tolerance
for scaling beyond single-machine limitations.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import Future
import threading
import queue
import json

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Try to import Ray, but gracefully handle if not available
try:
    import ray
    from ray.util.queue import Queue as RayQueue
    RAY_AVAILABLE = True
    logger.info("Ray distributed computing available")
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray not available - distributed features disabled")
    

@dataclass
class DistributedConfig:
    """Configuration for distributed computing."""
    
    # Ray cluster settings
    enable_distributed: bool = False
    ray_address: Optional[str] = None  # "ray://head-node-ip:10001" for cluster
    ray_runtime_env: Dict[str, Any] = field(default_factory=dict)
    
    # Worker settings
    num_sentiment_workers: int = 4
    num_contagion_workers: int = 2
    num_telemetry_workers: int = 1
    
    # Resource allocation
    worker_cpu_allocation: float = 1.0
    worker_memory_allocation: str = "1GB"
    
    # Fault tolerance
    enable_fault_tolerance: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Batch settings for distributed processing
    distributed_batch_size: int = 100
    max_queue_size: int = 1000
    worker_timeout: float = 30.0


class MockRay:
    """Mock Ray implementation for testing when Ray is not available."""
    
    @staticmethod
    def init(**kwargs):
        logger.info("MockRay: Simulating Ray initialization")
        return True
    
    @staticmethod
    def shutdown():
        logger.info("MockRay: Simulating Ray shutdown")
    
    @staticmethod
    def remote(func):
        """Mock remote decorator that just returns the function."""
        def wrapper(*args, **kwargs):
            return MockRemoteFunction(func, args, kwargs)
        wrapper.remote = lambda *args, **kwargs: MockRemoteFunction(func, args, kwargs)
        return wrapper
    
    @staticmethod
    def get(futures):
        """Mock ray.get that returns results immediately."""
        if isinstance(futures, list):
            return [future.get() for future in futures]
        else:
            return futures.get()
    
    @staticmethod
    def put(obj):
        """Mock ray.put that just returns the object."""
        return obj
    
    @staticmethod
    def wait(futures, num_returns=1):
        """Mock ray.wait that returns all futures as ready."""
        return futures[:num_returns], futures[num_returns:]


class MockRemoteFunction:
    """Mock remote function result."""
    
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._result = None
        self._computed = False
    
    def get(self):
        if not self._computed:
            self._result = self.func(*self.args, **self.kwargs)
            self._computed = True
        return self._result


# Use Ray if available, otherwise use mock
if RAY_AVAILABLE:
    ray_module = ray
else:
    ray_module = MockRay()


@ray_module.remote
class SentimentWorker:
    """Distributed worker for sentiment analysis."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.processed_count = 0
        self.start_time = time.time()
        
        # Initialize sentiment processor
        from ..sentiment.processor import SentimentProcessor
        self.processor = SentimentProcessor()
        
        logger.info(f"SentimentWorker {worker_id} initialized")
    
    def process_text_sentiments(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process batch of text sentiments."""
        results = []
        
        for text in texts:
            try:
                sentiment_data = self.processor.analyze_text_sentiment(text)
                results.append({
                    'polarity': sentiment_data.polarity.value,
                    'intensity': sentiment_data.intensity,
                    'confidence': sentiment_data.confidence,
                    'emotional_dimensions': sentiment_data.emotional_dimensions,
                    'processing_time': sentiment_data.processing_time
                })
                self.processed_count += 1
                
            except Exception as e:
                logger.error(f"SentimentWorker {self.worker_id} text processing error: {e}")
                results.append({
                    'polarity': 'NEUTRAL',
                    'intensity': 0.0,
                    'confidence': 0.0,
                    'emotional_dimensions': {},
                    'processing_time': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def process_behavioral_sentiments(self, behavioral_data: List[Tuple[List[int], Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Process batch of behavioral sentiments."""
        results = []
        
        for actions, context in behavioral_data:
            try:
                sentiment_data = self.processor.analyze_behavioral_sentiment(actions, context)
                results.append({
                    'polarity': sentiment_data.polarity.value,
                    'intensity': sentiment_data.intensity,
                    'confidence': sentiment_data.confidence,
                    'emotional_dimensions': sentiment_data.emotional_dimensions,
                    'processing_time': sentiment_data.processing_time
                })
                self.processed_count += 1
                
            except Exception as e:
                logger.error(f"SentimentWorker {self.worker_id} behavioral processing error: {e}")
                results.append({
                    'polarity': 'NEUTRAL',
                    'intensity': 0.0,
                    'confidence': 0.0,
                    'emotional_dimensions': {},
                    'processing_time': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        uptime = time.time() - self.start_time
        return {
            'worker_id': self.worker_id,
            'processed_count': self.processed_count,
            'uptime_seconds': uptime,
            'processing_rate': self.processed_count / max(uptime, 1.0)
        }


@ray_module.remote  
class ContagionWorker:
    """Distributed worker for emotional contagion processing."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.processed_count = 0
        self.start_time = time.time()
        
        # Initialize contagion system
        from ..sentiment.contagion import SentimentContagion, ContagionParameters
        params = ContagionParameters()
        self.contagion = SentimentContagion(params)
        
        logger.info(f"ContagionWorker {worker_id} initialized")
    
    def process_contagion_batch(self, agent_emotional_states: Dict[int, Dict[str, float]], 
                               agent_positions: Dict[int, List[float]]) -> Dict[int, Dict[str, float]]:
        """Process emotional contagion for a batch of agents."""
        try:
            # Convert to proper format for processing
            from ..sentiment.emotional_state import EmotionalState
            
            emotional_states = {}
            for agent_id, state_dict in agent_emotional_states.items():
                state = EmotionalState(
                    agent_id=agent_id,
                    initial_arousal=state_dict.get('arousal', 0.0),
                    initial_valence=state_dict.get('valence', 0.0),
                    initial_dominance=state_dict.get('dominance', 0.0)
                )
                emotional_states[agent_id] = state
            
            # Process contagion
            influences = self.contagion.process_emotional_contagion(emotional_states, agent_positions)
            
            # Convert results back to serializable format
            result = {}
            for agent_id, state in emotional_states.items():
                result[agent_id] = {
                    'arousal': state.arousal,
                    'valence': state.valence,
                    'dominance': state.dominance
                }
            
            self.processed_count += len(agent_emotional_states)
            return result
            
        except Exception as e:
            logger.error(f"ContagionWorker {self.worker_id} processing error: {e}")
            return agent_emotional_states  # Return unchanged on error
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        uptime = time.time() - self.start_time
        return {
            'worker_id': self.worker_id,
            'processed_count': self.processed_count,
            'uptime_seconds': uptime,
            'processing_rate': self.processed_count / max(uptime, 1.0)
        }


@ray_module.remote
class TelemetryWorker:
    """Distributed worker for telemetry processing and aggregation."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.processed_count = 0
        self.start_time = time.time()
        
        # Initialize telemetry collector
        from ..monitoring.sentiment_telemetry import SentimentTelemetryCollector
        self.collector = SentimentTelemetryCollector()
        
        logger.info(f"TelemetryWorker {worker_id} initialized")
    
    def process_telemetry_batch(self, telemetry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process and aggregate batch of telemetry data."""
        try:
            aggregated_metrics = {
                'total_agents': 0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'average_intensity': 0.0,
                'emotional_contagion_events': 0,
                'processing_time_ms': 0.0
            }
            
            for data in telemetry_data:
                aggregated_metrics['total_agents'] += data.get('agent_count', 0)
                
                # Aggregate sentiment distribution
                sentiment_dist = data.get('sentiment_distribution', {})
                for sentiment, count in sentiment_dist.items():
                    if sentiment in aggregated_metrics['sentiment_distribution']:
                        aggregated_metrics['sentiment_distribution'][sentiment] += count
                
                aggregated_metrics['average_intensity'] += data.get('average_intensity', 0.0)
                aggregated_metrics['emotional_contagion_events'] += data.get('contagion_events', 0)
                aggregated_metrics['processing_time_ms'] += data.get('processing_time_ms', 0.0)
            
            # Calculate averages
            num_batches = len(telemetry_data)
            if num_batches > 0:
                aggregated_metrics['average_intensity'] /= num_batches
                aggregated_metrics['processing_time_ms'] /= num_batches
            
            self.processed_count += len(telemetry_data)
            return aggregated_metrics
            
        except Exception as e:
            logger.error(f"TelemetryWorker {self.worker_id} processing error: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        uptime = time.time() - self.start_time
        return {
            'worker_id': self.worker_id,
            'processed_count': self.processed_count,
            'uptime_seconds': uptime,
            'processing_rate': self.processed_count / max(uptime, 1.0)
        }


class DistributedSentimentManager:
    """Manages distributed sentiment processing across multiple workers."""
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        
        self.sentiment_workers = []
        self.contagion_workers = []
        self.telemetry_workers = []
        
        self.is_initialized = False
        self.is_cluster_mode = False
        
        # Task queues for load balancing
        self.sentiment_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.contagion_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.telemetry_queue = queue.Queue(maxsize=self.config.max_queue_size)
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0
        }
        
        logger.info(f"DistributedSentimentManager initialized with config: {self.config}")
    
    def initialize(self) -> bool:
        """Initialize distributed computing environment."""
        if not self.config.enable_distributed:
            logger.info("Distributed computing disabled")
            return False
        
        if not RAY_AVAILABLE:
            logger.warning("Ray not available - cannot initialize distributed computing")
            return False
        
        try:
            # Initialize Ray
            if self.config.ray_address:
                ray_module.init(
                    address=self.config.ray_address,
                    runtime_env=self.config.ray_runtime_env
                )
                self.is_cluster_mode = True
                logger.info(f"Connected to Ray cluster at {self.config.ray_address}")
            else:
                ray_module.init(
                    runtime_env=self.config.ray_runtime_env
                )
                logger.info("Initialized local Ray cluster")
            
            # Create workers
            self._create_workers()
            
            self.is_initialized = True
            logger.info("Distributed sentiment processing initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed computing: {e}")
            return False
    
    def _create_workers(self):
        """Create distributed workers."""
        try:
            # Create sentiment workers
            for i in range(self.config.num_sentiment_workers):
                worker = SentimentWorker.options(
                    num_cpus=self.config.worker_cpu_allocation,
                    memory=self.config.worker_memory_allocation
                ).remote(i)
                self.sentiment_workers.append(worker)
            
            # Create contagion workers  
            for i in range(self.config.num_contagion_workers):
                worker = ContagionWorker.options(
                    num_cpus=self.config.worker_cpu_allocation,
                    memory=self.config.worker_memory_allocation
                ).remote(i)
                self.contagion_workers.append(worker)
            
            # Create telemetry workers
            for i in range(self.config.num_telemetry_workers):
                worker = TelemetryWorker.options(
                    num_cpus=self.config.worker_cpu_allocation,
                    memory=self.config.worker_memory_allocation
                ).remote(i)
                self.telemetry_workers.append(worker)
            
            logger.info(f"Created {len(self.sentiment_workers)} sentiment workers, "
                       f"{len(self.contagion_workers)} contagion workers, "
                       f"{len(self.telemetry_workers)} telemetry workers")
            
        except Exception as e:
            logger.error(f"Failed to create workers: {e}")
            raise
    
    def process_text_sentiments_distributed(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process text sentiments using distributed workers."""
        if not self.is_initialized or not self.sentiment_workers:
            logger.warning("Distributed processing not available, falling back to local")
            return self._process_text_sentiments_local(texts)
        
        try:
            start_time = time.time()
            
            # Split texts into batches for workers
            batch_size = max(1, len(texts) // len(self.sentiment_workers))
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            
            # Submit tasks to workers
            futures = []
            for i, batch in enumerate(batches):
                if batch:  # Only submit non-empty batches
                    worker_idx = i % len(self.sentiment_workers)
                    future = self.sentiment_workers[worker_idx].process_text_sentiments.remote(batch)
                    futures.append(future)
            
            # Collect results
            results = []
            if futures:
                batch_results = ray_module.get(futures)
                for batch_result in batch_results:
                    results.extend(batch_result)
            
            processing_time = (time.time() - start_time) * 1000
            self.stats['tasks_submitted'] += len(texts)
            self.stats['tasks_completed'] += len(results)
            self.stats['total_processing_time'] += processing_time
            
            logger.debug(f"Processed {len(texts)} text sentiments in {processing_time:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Distributed text sentiment processing failed: {e}")
            self.stats['tasks_failed'] += len(texts)
            return self._process_text_sentiments_local(texts)
    
    def process_contagion_distributed(self, agent_states: Dict[int, Dict[str, float]], 
                                    agent_positions: Dict[int, List[float]]) -> Dict[int, Dict[str, float]]:
        """Process emotional contagion using distributed workers."""
        if not self.is_initialized or not self.contagion_workers:
            logger.warning("Distributed contagion processing not available")
            return agent_states  # Return unchanged
        
        try:
            start_time = time.time()
            
            # For now, use single worker (can be optimized for spatial partitioning later)
            worker = self.contagion_workers[0]
            future = worker.process_contagion_batch.remote(agent_states, agent_positions)
            result = ray_module.get(future)
            
            processing_time = (time.time() - start_time) * 1000
            self.stats['tasks_submitted'] += len(agent_states)
            self.stats['tasks_completed'] += len(result)
            self.stats['total_processing_time'] += processing_time
            
            logger.debug(f"Processed contagion for {len(agent_states)} agents in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Distributed contagion processing failed: {e}")
            self.stats['tasks_failed'] += len(agent_states)
            return agent_states
    
    def _process_text_sentiments_local(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Fallback local processing for text sentiments."""
        from ..sentiment.processor import SentimentProcessor
        
        processor = SentimentProcessor()
        results = []
        
        for text in texts:
            try:
                sentiment_data = processor.analyze_text_sentiment(text)
                results.append({
                    'polarity': sentiment_data.polarity.value,
                    'intensity': sentiment_data.intensity,
                    'confidence': sentiment_data.confidence,
                    'emotional_dimensions': sentiment_data.emotional_dimensions,
                    'processing_time': sentiment_data.processing_time
                })
            except Exception as e:
                logger.error(f"Local text processing error: {e}")
                results.append({
                    'polarity': 'NEUTRAL',
                    'intensity': 0.0,
                    'confidence': 0.0,
                    'emotional_dimensions': {},
                    'processing_time': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def get_worker_stats(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get statistics from all workers."""
        if not self.is_initialized:
            return {'sentiment_workers': [], 'contagion_workers': [], 'telemetry_workers': []}
        
        try:
            stats = {
                'sentiment_workers': [],
                'contagion_workers': [],
                'telemetry_workers': []
            }
            
            # Get sentiment worker stats
            if self.sentiment_workers:
                futures = [worker.get_stats.remote() for worker in self.sentiment_workers]
                stats['sentiment_workers'] = ray_module.get(futures)
            
            # Get contagion worker stats
            if self.contagion_workers:
                futures = [worker.get_stats.remote() for worker in self.contagion_workers]
                stats['contagion_workers'] = ray_module.get(futures)
            
            # Get telemetry worker stats
            if self.telemetry_workers:
                futures = [worker.get_stats.remote() for worker in self.telemetry_workers]
                stats['telemetry_workers'] = ray_module.get(futures)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get worker stats: {e}")
            return {'sentiment_workers': [], 'contagion_workers': [], 'telemetry_workers': []}
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the Ray cluster."""
        if not self.is_initialized or not RAY_AVAILABLE:
            return {'status': 'not_initialized'}
        
        try:
            cluster_resources = ray_module.cluster_resources()
            return {
                'status': 'active',
                'cluster_mode': self.is_cluster_mode,
                'available_resources': cluster_resources,
                'num_nodes': len(ray_module.nodes()) if hasattr(ray_module, 'nodes') else 1,
                'total_workers': (len(self.sentiment_workers) + 
                                len(self.contagion_workers) + 
                                len(self.telemetry_workers))
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if stats['tasks_submitted'] > 0:
            stats['success_rate'] = stats['tasks_completed'] / stats['tasks_submitted']
            stats['failure_rate'] = stats['tasks_failed'] / stats['tasks_submitted']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        if stats['tasks_completed'] > 0:
            stats['avg_processing_time_ms'] = stats['total_processing_time'] / stats['tasks_completed']
        else:
            stats['avg_processing_time_ms'] = 0.0
        
        # Add cluster information
        stats['cluster_info'] = self.get_cluster_info()
        
        return stats
    
    def shutdown(self):
        """Shutdown distributed computing environment."""
        try:
            if self.is_initialized and RAY_AVAILABLE:
                ray_module.shutdown()
                logger.info("Ray cluster shutdown complete")
            
            self.is_initialized = False
            self.sentiment_workers.clear()
            self.contagion_workers.clear()
            self.telemetry_workers.clear()
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")