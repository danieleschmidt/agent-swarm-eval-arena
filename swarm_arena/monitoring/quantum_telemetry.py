"""
Quantum-Enhanced Telemetry and Monitoring System.

Advanced monitoring system with quantum-inspired anomaly detection,
real-time performance analytics, and predictive failure detection.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta
import websockets
import threading
from queue import Queue
import logging
from pathlib import Path

from ..utils.logging import get_logger
from ..exceptions import MonitoringError

logger = get_logger(__name__)


@dataclass
class TelemetryEvent:
    """Structured telemetry event."""
    
    timestamp: float
    event_type: str
    source: str
    data: Dict[str, Any]
    severity: str = "info"  # debug, info, warning, error, critical
    correlation_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    agent_metrics: Dict[str, Any] = field(default_factory=dict)
    quantum_metrics: Dict[str, float] = field(default_factory=dict)
    
    
@dataclass
class AnomalyAlert:
    """Anomaly detection alert."""
    
    timestamp: float
    anomaly_type: str
    severity: str
    description: str
    affected_components: List[str]
    confidence_score: float
    suggested_actions: List[str]
    correlation_data: Dict[str, Any] = field(default_factory=dict)


class QuantumAnomalyDetector:
    """Quantum-inspired anomaly detection system."""
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 detection_threshold: float = 0.95,
                 memory_size: int = 1000):
        self.learning_rate = learning_rate
        self.detection_threshold = detection_threshold
        self.memory_size = memory_size
        
        # Quantum-inspired state tracking
        self.quantum_states = {}
        self.baseline_metrics = {}
        self.anomaly_history = []
        
        # Performance tracking
        self.metric_history = []
        self.performance_baselines = {}
        
    def update_baseline(self, metrics: PerformanceMetrics) -> None:
        """Update performance baselines using quantum-inspired learning."""
        try:
            current_time = metrics.timestamp
            
            # Extract key metrics
            key_metrics = {
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'gpu_usage': metrics.gpu_usage,
                'agent_count': metrics.agent_metrics.get('active_agents', 0),
                'step_time': metrics.agent_metrics.get('average_step_time', 0)
            }
            
            # Update quantum states (simplified quantum learning)
            for metric_name, value in key_metrics.items():
                if metric_name not in self.quantum_states:
                    self.quantum_states[metric_name] = {
                        'amplitude': 0.5,
                        'phase': 0.0,
                        'variance': 1.0,
                        'coherence': 1.0
                    }
                
                state = self.quantum_states[metric_name]
                
                # Quantum state evolution
                if metric_name in self.baseline_metrics:
                    baseline = self.baseline_metrics[metric_name]
                    deviation = abs(value - baseline) / (baseline + 1e-6)
                    
                    # Update amplitude based on deviation
                    state['amplitude'] += self.learning_rate * (0.5 - deviation)
                    state['amplitude'] = np.clip(state['amplitude'], 0.1, 1.0)
                    
                    # Update phase
                    state['phase'] += self.learning_rate * deviation
                    state['phase'] = state['phase'] % (2 * np.pi)
                    
                    # Update variance
                    state['variance'] = 0.9 * state['variance'] + 0.1 * (deviation ** 2)
                
                # Update baseline with exponential moving average
                if metric_name in self.baseline_metrics:
                    self.baseline_metrics[metric_name] = (
                        0.95 * self.baseline_metrics[metric_name] + 0.05 * value
                    )
                else:
                    self.baseline_metrics[metric_name] = value
            
            # Store metric history (bounded)
            self.metric_history.append(metrics)
            if len(self.metric_history) > self.memory_size:
                self.metric_history = self.metric_history[-self.memory_size//2:]
                
        except Exception as e:
            logger.error(f"Failed to update baseline: {e}")
    
    def detect_anomalies(self, metrics: PerformanceMetrics) -> List[AnomalyAlert]:
        """Detect anomalies using quantum-inspired algorithms."""
        anomalies = []
        
        try:
            current_time = metrics.timestamp
            
            # Check each metric for anomalies
            key_metrics = {
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'gpu_usage': metrics.gpu_usage,
                'agent_count': metrics.agent_metrics.get('active_agents', 0),
                'step_time': metrics.agent_metrics.get('average_step_time', 0)
            }
            
            for metric_name, value in key_metrics.items():
                if metric_name in self.baseline_metrics and metric_name in self.quantum_states:
                    baseline = self.baseline_metrics[metric_name]
                    state = self.quantum_states[metric_name]
                    
                    # Calculate quantum-inspired anomaly score
                    deviation = abs(value - baseline) / (baseline + 1e-6)
                    
                    # Quantum interference calculation
                    phase_factor = np.cos(state['phase'])
                    amplitude_factor = state['amplitude']
                    coherence_factor = state['coherence']
                    
                    anomaly_score = (
                        deviation * amplitude_factor * 
                        abs(phase_factor) * coherence_factor
                    )
                    
                    # Threshold-based detection
                    if anomaly_score > self.detection_threshold:
                        severity = self._determine_severity(anomaly_score, deviation)
                        
                        anomaly = AnomalyAlert(
                            timestamp=current_time,
                            anomaly_type=f"{metric_name}_anomaly",
                            severity=severity,
                            description=f"Anomalous {metric_name}: {value:.3f} (baseline: {baseline:.3f})",
                            affected_components=[metric_name],
                            confidence_score=min(anomaly_score, 1.0),
                            suggested_actions=self._get_suggested_actions(metric_name, deviation),
                            correlation_data={
                                'quantum_score': anomaly_score,
                                'deviation': deviation,
                                'baseline': baseline,
                                'current_value': value
                            }
                        )
                        
                        anomalies.append(anomaly)
            
            # Pattern-based anomaly detection
            pattern_anomalies = self._detect_pattern_anomalies(metrics)
            anomalies.extend(pattern_anomalies)
            
            # Update anomaly history
            for anomaly in anomalies:
                self.anomaly_history.append(anomaly)
            
            # Maintain bounded history
            if len(self.anomaly_history) > self.memory_size:
                self.anomaly_history = self.anomaly_history[-self.memory_size//2:]
                
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    def _determine_severity(self, anomaly_score: float, deviation: float) -> str:
        """Determine anomaly severity."""
        if anomaly_score > 2.0 or deviation > 0.5:
            return "critical"
        elif anomaly_score > 1.5 or deviation > 0.3:
            return "error"
        elif anomaly_score > 1.2 or deviation > 0.2:
            return "warning"
        else:
            return "info"
    
    def _get_suggested_actions(self, metric_name: str, deviation: float) -> List[str]:
        """Get suggested actions for metric anomalies."""
        actions = []
        
        if metric_name == "cpu_usage" and deviation > 0.2:
            actions.extend([
                "Check for CPU-intensive processes",
                "Consider scaling horizontally",
                "Optimize algorithm efficiency"
            ])
        elif metric_name == "memory_usage" and deviation > 0.2:
            actions.extend([
                "Check for memory leaks",
                "Optimize data structures",
                "Implement garbage collection"
            ])
        elif metric_name == "step_time" and deviation > 0.3:
            actions.extend([
                "Profile performance bottlenecks",
                "Optimize simulation algorithms",
                "Consider parallel processing"
            ])
        elif metric_name == "agent_count" and deviation > 0.1:
            actions.extend([
                "Check agent lifecycle management",
                "Investigate agent termination causes",
                "Review simulation parameters"
            ])
        
        if not actions:
            actions.append("Monitor for continued anomalous behavior")
        
        return actions
    
    def _detect_pattern_anomalies(self, metrics: PerformanceMetrics) -> List[AnomalyAlert]:
        """Detect pattern-based anomalies."""
        anomalies = []
        
        try:
            if len(self.metric_history) < 10:
                return anomalies
            
            # Check for sudden drops in performance
            recent_metrics = self.metric_history[-10:]
            step_times = [m.agent_metrics.get('average_step_time', 0) for m in recent_metrics]
            
            if len(step_times) >= 5:
                recent_avg = np.mean(step_times[-5:])
                historical_avg = np.mean(step_times[:-5])
                
                if recent_avg > historical_avg * 1.5:  # 50% increase
                    anomaly = AnomalyAlert(
                        timestamp=metrics.timestamp,
                        anomaly_type="performance_degradation",
                        severity="warning",
                        description=f"Performance degradation detected: {recent_avg:.3f}ms vs {historical_avg:.3f}ms",
                        affected_components=["simulation_engine"],
                        confidence_score=0.8,
                        suggested_actions=[
                            "Check system resources",
                            "Profile simulation performance",
                            "Consider reducing simulation complexity"
                        ]
                    )
                    anomalies.append(anomaly)
            
            # Check for memory growth patterns
            memory_usage = [m.memory_usage for m in recent_metrics]
            if len(memory_usage) >= 5:
                # Linear regression to detect memory growth
                x = np.arange(len(memory_usage))
                slope = np.polyfit(x, memory_usage, 1)[0]
                
                if slope > 0.05:  # 5% growth per measurement
                    anomaly = AnomalyAlert(
                        timestamp=metrics.timestamp,
                        anomaly_type="memory_leak",
                        severity="error",
                        description=f"Potential memory leak detected: growth rate {slope:.3f}%",
                        affected_components=["memory_management"],
                        confidence_score=0.7,
                        suggested_actions=[
                            "Check for memory leaks",
                            "Review object lifecycle management",
                            "Implement memory profiling"
                        ]
                    )
                    anomalies.append(anomaly)
                    
        except Exception as e:
            logger.error(f"Pattern anomaly detection failed: {e}")
        
        return anomalies


class QuantumTelemetryCollector:
    """Advanced telemetry collection with quantum-inspired analytics."""
    
    def __init__(self, 
                 buffer_size: int = 10000,
                 flush_interval: float = 5.0,
                 anomaly_detection: bool = True):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.anomaly_detection_enabled = anomaly_detection
        
        # Data structures
        self.event_buffer = Queue(maxsize=buffer_size)
        self.metrics_buffer = Queue(maxsize=buffer_size)
        self.subscribers = []
        
        # Anomaly detection
        if anomaly_detection:
            self.anomaly_detector = QuantumAnomalyDetector()
        
        # Processing threads
        self.processing_thread = None
        self.streaming_thread = None
        self.running = False
        
        # WebSocket server for real-time streaming
        self.websocket_server = None
        self.websocket_clients = set()
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'metrics_processed': 0,
            'anomalies_detected': 0,
            'start_time': time.time()
        }
    
    def start(self, websocket_port: int = 8765) -> None:
        """Start telemetry collection and processing."""
        try:
            self.running = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._process_telemetry,
                daemon=True
            )
            self.processing_thread.start()
            
            # Start WebSocket server
            if websocket_port:
                self.streaming_thread = threading.Thread(
                    target=self._start_websocket_server,
                    args=(websocket_port,),
                    daemon=True
                )
                self.streaming_thread.start()
            
            logger.info(f"Quantum telemetry collector started on port {websocket_port}")
            
        except Exception as e:
            logger.error(f"Failed to start telemetry collector: {e}")
            raise MonitoringError(f"Telemetry start failed: {e}")
    
    def stop(self) -> None:
        """Stop telemetry collection."""
        try:
            self.running = False
            
            if self.websocket_server:
                self.websocket_server.close()
            
            logger.info("Quantum telemetry collector stopped")
            
        except Exception as e:
            logger.error(f"Error stopping telemetry collector: {e}")
    
    def collect_event(self, event: TelemetryEvent) -> None:
        """Collect a telemetry event."""
        try:
            if not self.event_buffer.full():
                self.event_buffer.put(event)
                self.stats['events_processed'] += 1
            else:
                logger.warning("Event buffer full, dropping event")
                
        except Exception as e:
            logger.error(f"Failed to collect event: {e}")
    
    def collect_metrics(self, metrics: PerformanceMetrics) -> None:
        """Collect performance metrics."""
        try:
            if not self.metrics_buffer.full():
                self.metrics_buffer.put(metrics)
                self.stats['metrics_processed'] += 1
                
                # Update anomaly detection baseline
                if self.anomaly_detection_enabled:
                    self.anomaly_detector.update_baseline(metrics)
            else:
                logger.warning("Metrics buffer full, dropping metrics")
                
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    def add_subscriber(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a telemetry subscriber."""
        self.subscribers.append(callback)
    
    def _process_telemetry(self) -> None:
        """Process telemetry data in background thread."""
        last_flush = time.time()
        
        while self.running:
            try:
                # Process events
                events_batch = []
                while not self.event_buffer.empty() and len(events_batch) < 100:
                    try:
                        event = self.event_buffer.get_nowait()
                        events_batch.append(event)
                    except:
                        break
                
                # Process metrics
                metrics_batch = []
                while not self.metrics_buffer.empty() and len(metrics_batch) < 100:
                    try:
                        metrics = self.metrics_buffer.get_nowait()
                        metrics_batch.append(metrics)
                    except:
                        break
                
                # Anomaly detection on latest metrics
                anomalies = []
                if metrics_batch and self.anomaly_detection_enabled:
                    latest_metrics = metrics_batch[-1]
                    detected_anomalies = self.anomaly_detector.detect_anomalies(latest_metrics)
                    anomalies.extend(detected_anomalies)
                    self.stats['anomalies_detected'] += len(detected_anomalies)
                
                # Notify subscribers
                if events_batch or metrics_batch or anomalies:
                    telemetry_data = {
                        'timestamp': time.time(),
                        'events': [self._serialize_event(e) for e in events_batch],
                        'metrics': [self._serialize_metrics(m) for m in metrics_batch],
                        'anomalies': [self._serialize_anomaly(a) for a in anomalies],
                        'stats': self.stats.copy()
                    }
                    
                    # Notify all subscribers
                    for subscriber in self.subscribers:
                        try:
                            subscriber(telemetry_data)
                        except Exception as e:
                            logger.error(f"Subscriber notification failed: {e}")
                    
                    # Stream to WebSocket clients
                    if self.websocket_clients:
                        asyncio.run_coroutine_threadsafe(
                            self._broadcast_to_websockets(telemetry_data),
                            asyncio.get_event_loop()
                        )
                
                # Periodic flush
                current_time = time.time()
                if current_time - last_flush > self.flush_interval:
                    self._flush_data()
                    last_flush = current_time
                
                time.sleep(0.1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Telemetry processing error: {e}")
                time.sleep(1.0)
    
    def _start_websocket_server(self, port: int) -> None:
        """Start WebSocket server for real-time streaming."""
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
            
            async def handle_client(websocket, path):
                self.websocket_clients.add(websocket)
                try:
                    await websocket.wait_closed()
                finally:
                    self.websocket_clients.discard(websocket)
            
            start_server = websockets.serve(handle_client, "localhost", port)
            
            loop = asyncio.get_event_loop()
            self.websocket_server = loop.run_until_complete(start_server)
            
            logger.info(f"WebSocket telemetry server started on port {port}")
            loop.run_forever()
            
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
    
    async def _broadcast_to_websockets(self, data: Dict[str, Any]) -> None:
        """Broadcast telemetry data to WebSocket clients."""
        if not self.websocket_clients:
            return
        
        message = json.dumps(data, default=str)
        disconnected_clients = set()
        
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"WebSocket broadcast error: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    def _serialize_event(self, event: TelemetryEvent) -> Dict[str, Any]:
        """Serialize telemetry event to dictionary."""
        return {
            'timestamp': event.timestamp,
            'event_type': event.event_type,
            'source': event.source,
            'data': event.data,
            'severity': event.severity,
            'correlation_id': event.correlation_id,
            'tags': event.tags
        }
    
    def _serialize_metrics(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Serialize performance metrics to dictionary."""
        return {
            'timestamp': metrics.timestamp,
            'cpu_usage': metrics.cpu_usage,
            'memory_usage': metrics.memory_usage,
            'gpu_usage': metrics.gpu_usage,
            'network_io': metrics.network_io,
            'agent_metrics': metrics.agent_metrics,
            'quantum_metrics': metrics.quantum_metrics
        }
    
    def _serialize_anomaly(self, anomaly: AnomalyAlert) -> Dict[str, Any]:
        """Serialize anomaly alert to dictionary."""
        return {
            'timestamp': anomaly.timestamp,
            'anomaly_type': anomaly.anomaly_type,
            'severity': anomaly.severity,
            'description': anomaly.description,
            'affected_components': anomaly.affected_components,
            'confidence_score': anomaly.confidence_score,
            'suggested_actions': anomaly.suggested_actions,
            'correlation_data': anomaly.correlation_data
        }
    
    def _flush_data(self) -> None:
        """Flush collected data to persistent storage."""
        try:
            # Create output directory
            output_dir = Path("telemetry_data")
            output_dir.mkdir(exist_ok=True)
            
            # Save statistics
            stats_file = output_dir / f"telemetry_stats_{int(time.time())}.json"
            with open(stats_file, 'w') as f:
                stats_data = self.stats.copy()
                stats_data['uptime'] = time.time() - stats_data['start_time']
                json.dump(stats_data, f, indent=2)
            
            logger.debug(f"Telemetry data flushed to {stats_file}")
            
        except Exception as e:
            logger.error(f"Failed to flush telemetry data: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get telemetry collection statistics."""
        stats = self.stats.copy()
        stats['uptime'] = time.time() - stats['start_time']
        stats['buffer_utilization'] = {
            'events': self.event_buffer.qsize() / self.buffer_size,
            'metrics': self.metrics_buffer.qsize() / self.buffer_size
        }
        stats['websocket_clients'] = len(self.websocket_clients)
        
        if self.anomaly_detection_enabled:
            stats['anomaly_detection'] = {
                'baseline_metrics': len(self.anomaly_detector.baseline_metrics),
                'quantum_states': len(self.anomaly_detector.quantum_states),
                'anomaly_history_size': len(self.anomaly_detector.anomaly_history)
            }
        
        return stats


class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, telemetry_collector: QuantumTelemetryCollector):
        self.telemetry = telemetry_collector
        self.health_checks = {}
        self.health_status = "healthy"
        self.last_check = time.time()
        
    def add_health_check(self, name: str, check_function: Callable[[], bool], 
                        interval: float = 30.0) -> None:
        """Add a health check function."""
        self.health_checks[name] = {
            'function': check_function,
            'interval': interval,
            'last_run': 0,
            'status': 'unknown'
        }
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return status."""
        current_time = time.time()
        health_report = {
            'timestamp': current_time,
            'overall_status': 'healthy',
            'checks': {}
        }
        
        failed_checks = 0
        
        for name, check_info in self.health_checks.items():
            if current_time - check_info['last_run'] >= check_info['interval']:
                try:
                    result = check_info['function']()
                    check_info['status'] = 'pass' if result else 'fail'
                    check_info['last_run'] = current_time
                    
                    if not result:
                        failed_checks += 1
                        
                except Exception as e:
                    check_info['status'] = 'error'
                    check_info['error'] = str(e)
                    failed_checks += 1
                    
            health_report['checks'][name] = {
                'status': check_info['status'],
                'last_run': check_info['last_run']
            }
        
        # Determine overall health
        if failed_checks == 0:
            health_report['overall_status'] = 'healthy'
        elif failed_checks <= len(self.health_checks) * 0.3:
            health_report['overall_status'] = 'degraded'
        else:
            health_report['overall_status'] = 'unhealthy'
        
        self.health_status = health_report['overall_status']
        self.last_check = current_time
        
        # Send health report as telemetry event
        self.telemetry.collect_event(TelemetryEvent(
            timestamp=current_time,
            event_type="health_check",
            source="health_monitor",
            data=health_report,
            severity="info" if health_report['overall_status'] == 'healthy' else "warning"
        ))
        
        return health_report
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        # Run health checks if needed
        if time.time() - self.last_check > 60:  # Check at least every minute
            self.run_health_checks()
        
        return self.health_status == 'healthy'