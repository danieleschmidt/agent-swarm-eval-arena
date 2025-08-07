"""
Sentiment telemetry and monitoring for multi-agent systems.

Provides real-time sentiment tracking, emotional analytics, and
visualization capabilities for large-scale sentiment-aware simulations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import time
import json
from collections import defaultdict, deque
import asyncio
import websockets
import threading

from ..sentiment.emotional_state import EmotionalState, EmotionType
from ..sentiment.contagion import SentimentContagion
from ..monitoring.telemetry import TelemetryCollector, TelemetryData
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentMetrics:
    """Container for sentiment-related metrics."""
    
    timestamp: float = field(default_factory=time.time)
    
    # Individual agent metrics
    agent_emotions: Dict[int, Dict[str, float]] = field(default_factory=dict)
    agent_sentiment_scores: Dict[int, float] = field(default_factory=dict)
    
    # Population-level metrics
    population_arousal: float = 0.0
    population_valence: float = 0.0
    population_dominance: float = 0.0
    dominant_emotion: str = "neutral"
    emotional_diversity: float = 0.0
    
    # Contagion metrics
    active_influences: int = 0
    contagion_events: int = 0
    emotional_clusters: int = 0
    influence_network_density: float = 0.0
    
    # Performance metrics
    sentiment_processing_time: float = 0.0
    contagion_processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'agent_emotions': self.agent_emotions,
            'agent_sentiment_scores': self.agent_sentiment_scores,
            'population_metrics': {
                'arousal': self.population_arousal,
                'valence': self.population_valence,
                'dominance': self.population_dominance,
                'dominant_emotion': self.dominant_emotion,
                'emotional_diversity': self.emotional_diversity
            },
            'contagion_metrics': {
                'active_influences': self.active_influences,
                'contagion_events': self.contagion_events,
                'emotional_clusters': self.emotional_clusters,
                'influence_network_density': self.influence_network_density
            },
            'performance_metrics': {
                'sentiment_processing_time': self.sentiment_processing_time,
                'contagion_processing_time': self.contagion_processing_time
            }
        }


class SentimentTelemetryCollector:
    """
    Advanced telemetry collector for sentiment-aware multi-agent systems.
    
    Provides real-time sentiment monitoring, emotional analytics,
    and WebSocket streaming capabilities.
    """
    
    def __init__(self, buffer_size: int = 1000, streaming_enabled: bool = False, 
                 streaming_port: int = 8081):
        """
        Initialize sentiment telemetry collector.
        
        Args:
            buffer_size: Size of metrics buffer
            streaming_enabled: Enable WebSocket streaming
            streaming_port: Port for WebSocket server
        """
        self.buffer_size = buffer_size
        self.streaming_enabled = streaming_enabled
        self.streaming_port = streaming_port
        
        # Metrics storage
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.emotion_history = defaultdict(lambda: deque(maxlen=200))
        self.contagion_events_history = deque(maxlen=500)
        
        # Performance tracking
        self.total_updates = 0
        self.last_update_time = time.time()
        
        # WebSocket streaming
        self.websocket_clients = set()
        self.streaming_server = None
        self.streaming_thread = None
        
        # Analytics cache
        self.analytics_cache = {}
        self.cache_update_interval = 1.0  # seconds
        self.last_cache_update = 0.0
        
        if streaming_enabled:
            self._start_streaming_server()
        
        logger.info(f"SentimentTelemetryCollector initialized with buffer_size={buffer_size}")
    
    def collect_sentiment_metrics(self, agent_emotions: Dict[int, EmotionalState],
                                contagion_system: Optional[SentimentContagion] = None,
                                processing_times: Optional[Dict[str, float]] = None) -> SentimentMetrics:
        """
        Collect comprehensive sentiment metrics from agent population.
        
        Args:
            agent_emotions: Dictionary of agent emotional states
            contagion_system: Sentiment contagion system (optional)
            processing_times: Performance timing data (optional)
            
        Returns:
            SentimentMetrics object containing collected data
        """
        try:
            start_time = time.time()
            
            metrics = SentimentMetrics()
            
            if not agent_emotions:
                return metrics
            
            # Collect individual agent metrics
            arousal_values = []
            valence_values = []
            dominance_values = []
            emotion_counts = defaultdict(int)
            
            for agent_id, emotional_state in agent_emotions.items():
                # Get emotional dimensions
                dimensions = {
                    'arousal': emotional_state.arousal,
                    'valence': emotional_state.valence,
                    'dominance': emotional_state.dominance
                }
                
                metrics.agent_emotions[agent_id] = dimensions
                
                # Collect for population statistics
                arousal_values.append(emotional_state.arousal)
                valence_values.append(emotional_state.valence)
                dominance_values.append(emotional_state.dominance)
                
                # Get dominant emotion
                dominant_emotion, strength = emotional_state.get_dominant_emotion()
                emotion_counts[dominant_emotion.value] += 1
                
                # Calculate sentiment score (composite measure)
                sentiment_score = (emotional_state.valence + (1.0 - abs(emotional_state.arousal)) * 0.5) / 1.5
                metrics.agent_sentiment_scores[agent_id] = sentiment_score
                
                # Update emotion history for trending
                self.emotion_history[agent_id].append({
                    'timestamp': metrics.timestamp,
                    'arousal': emotional_state.arousal,
                    'valence': emotional_state.valence,
                    'dominance': emotional_state.dominance,
                    'dominant_emotion': dominant_emotion.value
                })
            
            # Calculate population-level metrics
            if arousal_values:
                metrics.population_arousal = float(np.mean(arousal_values))
                metrics.population_valence = float(np.mean(valence_values))
                metrics.population_dominance = float(np.mean(dominance_values))
                
                # Find population dominant emotion
                metrics.dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
                
                # Calculate emotional diversity (entropy-based)
                total_agents = len(agent_emotions)
                emotion_probs = [count / total_agents for count in emotion_counts.values()]
                emotion_probs = [p for p in emotion_probs if p > 0]  # Remove zeros
                
                if emotion_probs:
                    metrics.emotional_diversity = float(-np.sum(p * np.log2(p) for p in emotion_probs))
                else:
                    metrics.emotional_diversity = 0.0
            
            # Collect contagion metrics
            if contagion_system:
                contagion_stats = contagion_system.get_contagion_statistics()
                metrics.active_influences = contagion_stats.get('active_influences', 0)
                metrics.contagion_events = contagion_stats.get('contagion_events', 0)
                metrics.emotional_clusters = contagion_stats.get('emotional_clusters', 0)
                metrics.influence_network_density = contagion_stats.get('avg_connections_per_agent', 0.0)
                
                # Track contagion events
                if metrics.contagion_events > 0:
                    self.contagion_events_history.append({
                        'timestamp': metrics.timestamp,
                        'events': metrics.contagion_events,
                        'clusters': metrics.emotional_clusters,
                        'influences': metrics.active_influences
                    })
            
            # Add processing time metrics
            if processing_times:
                metrics.sentiment_processing_time = processing_times.get('sentiment', 0.0)
                metrics.contagion_processing_time = processing_times.get('contagion', 0.0)
            
            # Store metrics
            self.metrics_buffer.append(metrics)
            self.total_updates += 1
            self.last_update_time = time.time()
            
            # Stream to WebSocket clients if enabled
            if self.streaming_enabled and self.websocket_clients:
                asyncio.create_task(self._broadcast_metrics(metrics))
            
            collection_time = (time.time() - start_time) * 1000
            logger.debug(f"Collected sentiment metrics for {len(agent_emotions)} agents in {collection_time:.2f}ms")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Collecting sentiment metrics failed: {str(e)}")
            return SentimentMetrics()
    
    def get_sentiment_analytics(self, time_window: int = 60) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analytics over specified time window.
        
        Args:
            time_window: Time window in seconds for analysis
            
        Returns:
            Dictionary containing sentiment analytics
        """
        try:
            current_time = time.time()
            
            # Check cache
            if (current_time - self.last_cache_update) < self.cache_update_interval:
                return self.analytics_cache
            
            analytics = {
                'timestamp': current_time,
                'time_window': time_window,
                'total_updates': self.total_updates,
                'buffer_size': len(self.metrics_buffer),
                'population_trends': {},
                'emotion_distributions': {},
                'contagion_analysis': {},
                'agent_analytics': {},
                'performance_metrics': {}
            }
            
            # Filter metrics within time window
            window_start = current_time - time_window
            recent_metrics = [m for m in self.metrics_buffer if m.timestamp >= window_start]
            
            if not recent_metrics:
                return analytics
            
            # Population trends
            arousal_trend = [m.population_arousal for m in recent_metrics]
            valence_trend = [m.population_valence for m in recent_metrics]
            dominance_trend = [m.population_dominance for m in recent_metrics]
            
            analytics['population_trends'] = {
                'arousal': {
                    'current': arousal_trend[-1] if arousal_trend else 0.0,
                    'mean': float(np.mean(arousal_trend)) if arousal_trend else 0.0,
                    'std': float(np.std(arousal_trend)) if arousal_trend else 0.0,
                    'trend': self._calculate_trend(arousal_trend),
                    'values': arousal_trend[-20:]  # Last 20 values
                },
                'valence': {
                    'current': valence_trend[-1] if valence_trend else 0.0,
                    'mean': float(np.mean(valence_trend)) if valence_trend else 0.0,
                    'std': float(np.std(valence_trend)) if valence_trend else 0.0,
                    'trend': self._calculate_trend(valence_trend),
                    'values': valence_trend[-20:]
                },
                'dominance': {
                    'current': dominance_trend[-1] if dominance_trend else 0.0,
                    'mean': float(np.mean(dominance_trend)) if dominance_trend else 0.0,
                    'std': float(np.std(dominance_trend)) if dominance_trend else 0.0,
                    'trend': self._calculate_trend(dominance_trend),
                    'values': dominance_trend[-20:]
                }
            }
            
            # Emotion distributions
            emotion_distribution = defaultdict(int)
            diversity_values = []
            
            for metrics in recent_metrics:
                emotion_distribution[metrics.dominant_emotion] += 1
                diversity_values.append(metrics.emotional_diversity)
            
            total_samples = len(recent_metrics)
            analytics['emotion_distributions'] = {
                emotion: count / total_samples for emotion, count in emotion_distribution.items()
            }
            
            analytics['emotion_distributions']['diversity'] = {
                'current': diversity_values[-1] if diversity_values else 0.0,
                'mean': float(np.mean(diversity_values)) if diversity_values else 0.0,
                'trend': self._calculate_trend(diversity_values)
            }
            
            # Contagion analysis
            contagion_events = [m.contagion_events for m in recent_metrics]
            influence_counts = [m.active_influences for m in recent_metrics]
            cluster_counts = [m.emotional_clusters for m in recent_metrics]
            
            analytics['contagion_analysis'] = {
                'total_events': sum(contagion_events),
                'avg_influences': float(np.mean(influence_counts)) if influence_counts else 0.0,
                'avg_clusters': float(np.mean(cluster_counts)) if cluster_counts else 0.0,
                'contagion_rate': sum(contagion_events) / max(1, len(contagion_events)),
                'recent_events': list(self.contagion_events_history)[-10:]
            }
            
            # Agent-level analytics
            if recent_metrics:
                latest_metrics = recent_metrics[-1]
                agent_count = len(latest_metrics.agent_emotions)
                
                # Find most/least emotional agents
                if latest_metrics.agent_emotions:
                    agent_intensities = {}
                    for agent_id, emotions in latest_metrics.agent_emotions.items():
                        intensity = np.sqrt(emotions['arousal']**2 + emotions['valence']**2 + emotions['dominance']**2)
                        agent_intensities[agent_id] = intensity
                    
                    most_emotional = max(agent_intensities, key=agent_intensities.get) if agent_intensities else None
                    least_emotional = min(agent_intensities, key=agent_intensities.get) if agent_intensities else None
                    
                    analytics['agent_analytics'] = {
                        'total_agents': agent_count,
                        'most_emotional_agent': most_emotional,
                        'least_emotional_agent': least_emotional,
                        'avg_emotional_intensity': float(np.mean(list(agent_intensities.values()))),
                        'emotional_intensity_std': float(np.std(list(agent_intensities.values())))
                    }
            
            # Performance metrics
            processing_times = [m.sentiment_processing_time for m in recent_metrics if m.sentiment_processing_time > 0]
            contagion_times = [m.contagion_processing_time for m in recent_metrics if m.contagion_processing_time > 0]
            
            analytics['performance_metrics'] = {
                'avg_sentiment_processing_time': float(np.mean(processing_times)) if processing_times else 0.0,
                'avg_contagion_processing_time': float(np.mean(contagion_times)) if contagion_times else 0.0,
                'update_frequency': len(recent_metrics) / time_window,
                'last_update': self.last_update_time
            }
            
            # Cache results
            self.analytics_cache = analytics
            self.last_cache_update = current_time
            
            return analytics
            
        except Exception as e:
            logger.error(f"Getting sentiment analytics failed: {str(e)}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def get_emotion_field_data(self, agent_emotions: Dict[int, EmotionalState],
                              agent_positions: Dict[int, np.ndarray],
                              contagion_system: Optional[SentimentContagion] = None) -> Dict[str, Any]:
        """
        Get emotional field data for visualization.
        
        Args:
            agent_emotions: Dictionary of agent emotional states
            agent_positions: Dictionary of agent positions
            contagion_system: Sentiment contagion system (optional)
            
        Returns:
            Dictionary containing emotional field visualization data
        """
        try:
            field_data = {
                'timestamp': time.time(),
                'agent_data': [],
                'field_maps': {}
            }
            
            # Agent-level data for visualization
            for agent_id, emotional_state in agent_emotions.items():
                if agent_id not in agent_positions:
                    continue
                
                position = agent_positions[agent_id]
                dominant_emotion, strength = emotional_state.get_dominant_emotion()
                
                agent_data = {
                    'agent_id': agent_id,
                    'position': position.tolist() if hasattr(position, 'tolist') else list(position),
                    'arousal': emotional_state.arousal,
                    'valence': emotional_state.valence,
                    'dominance': emotional_state.dominance,
                    'dominant_emotion': dominant_emotion.value,
                    'emotion_strength': strength,
                    'color': self._emotion_to_color(dominant_emotion, strength)
                }
                
                field_data['agent_data'].append(agent_data)
            
            # Generate field maps if contagion system available
            if contagion_system and agent_positions:
                field_maps = contagion_system.get_emotional_field_map(
                    agent_emotions, agent_positions, grid_resolution=30
                )
                
                # Convert numpy arrays to lists for JSON serialization
                for field_name, field_array in field_maps.items():
                    if isinstance(field_array, np.ndarray):
                        field_data['field_maps'][field_name] = field_array.tolist()
                    else:
                        field_data['field_maps'][field_name] = field_array
            
            return field_data
            
        except Exception as e:
            logger.error(f"Getting emotion field data failed: {str(e)}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def _start_streaming_server(self) -> None:
        """Start WebSocket streaming server in separate thread."""
        try:
            def run_server():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def handle_client(websocket, path):
                    logger.info(f"New WebSocket client connected: {websocket.remote_address}")
                    self.websocket_clients.add(websocket)
                    
                    try:
                        await websocket.wait_closed()
                    finally:
                        self.websocket_clients.remove(websocket)
                        logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
                
                start_server = websockets.serve(handle_client, "localhost", self.streaming_port)
                logger.info(f"Sentiment telemetry WebSocket server started on port {self.streaming_port}")
                
                loop.run_until_complete(start_server)
                loop.run_forever()
            
            self.streaming_thread = threading.Thread(target=run_server, daemon=True)
            self.streaming_thread.start()
            
        except Exception as e:
            logger.error(f"Starting streaming server failed: {str(e)}")
            self.streaming_enabled = False
    
    async def _broadcast_metrics(self, metrics: SentimentMetrics) -> None:
        """Broadcast metrics to WebSocket clients."""
        try:
            if not self.websocket_clients:
                return
            
            message = json.dumps(metrics.to_dict())
            disconnected_clients = set()
            
            for client in self.websocket_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception as e:
                    logger.warning(f"Failed to send message to client: {str(e)}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            for client in disconnected_clients:
                self.websocket_clients.discard(client)
                
        except Exception as e:
            logger.error(f"Broadcasting metrics failed: {str(e)}")
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 5:
            return "insufficient_data"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        
        try:
            slope, _ = np.polyfit(x, y, 1)
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"
                
        except Exception:
            return "unknown"
    
    def _emotion_to_color(self, emotion: EmotionType, strength: float) -> str:
        """Convert emotion to color for visualization."""
        # Color mapping for different emotions
        emotion_colors = {
            EmotionType.JOY: "#FFD700",      # Gold
            EmotionType.ANGER: "#FF4500",    # Red-Orange
            EmotionType.SADNESS: "#4169E1",  # Royal Blue
            EmotionType.FEAR: "#8B008B",     # Dark Magenta
            EmotionType.SURPRISE: "#FF69B4", # Hot Pink
            EmotionType.DISGUST: "#228B22",  # Forest Green
            EmotionType.TRUST: "#00CED1",    # Dark Turquoise
            EmotionType.ANTICIPATION: "#FFA500", # Orange
            EmotionType.NEUTRAL: "#808080"   # Gray
        }
        
        base_color = emotion_colors.get(emotion, "#808080")
        
        # Adjust opacity based on strength
        opacity = max(0.3, min(1.0, strength))
        
        # Convert hex to RGBA
        hex_color = base_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"
    
    def clear_buffers(self) -> None:
        """Clear all metric buffers."""
        try:
            self.metrics_buffer.clear()
            self.emotion_history.clear()
            self.contagion_events_history.clear()
            self.analytics_cache.clear()
            
            logger.info("Sentiment telemetry buffers cleared")
            
        except Exception as e:
            logger.error(f"Clearing buffers failed: {str(e)}")
    
    def stop_streaming(self) -> None:
        """Stop WebSocket streaming server."""
        try:
            if self.streaming_server:
                self.streaming_server.close()
            
            self.streaming_enabled = False
            
            logger.info("Sentiment telemetry streaming stopped")
            
        except Exception as e:
            logger.error(f"Stopping streaming failed: {str(e)}")