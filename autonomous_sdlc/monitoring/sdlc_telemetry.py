"""
SDLC Telemetry Collector: Comprehensive telemetry collection
for autonomous SDLC execution monitoring and optimization.
"""

import time
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import psutil
import threading

@dataclass
class TelemetryEvent:
    """Represents a telemetry event."""
    timestamp: float
    event_type: str
    event_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "event_type": self.event_type,
            "event_name": self.event_name,
            "data": self.data,
            "duration": self.duration,
            "success": self.success,
            "error_message": self.error_message
        }

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int] = field(default_factory=dict)
    process_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "disk_usage_percent": self.disk_usage_percent,
            "network_io": self.network_io,
            "process_count": self.process_count
        }

class SDLCTelemetryCollector:
    """Comprehensive telemetry collector for autonomous SDLC."""
    
    def __init__(self, 
                 collection_interval: float = 5.0,
                 max_events: int = 10000):
        """Initialize telemetry collector.
        
        Args:
            collection_interval: Interval for system metrics collection (seconds)
            max_events: Maximum number of events to keep in memory
        """
        self.collection_interval = collection_interval
        self.max_events = max_events
        
        # Event storage
        self.events: List[TelemetryEvent] = []
        self.system_metrics: List[SystemMetrics] = []
        
        # Active tracking
        self.active_checkpoints: Dict[str, float] = {}  # checkpoint -> start_time
        self.active_generations: Dict[int, float] = {}  # generation -> start_time
        
        # Metrics aggregation
        self.checkpoint_metrics: Dict[str, List[float]] = {}  # checkpoint -> durations
        self.generation_metrics: Dict[int, Dict[str, Any]] = {}
        
        # System monitoring
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_lock = threading.Lock()
        
        # Export configuration
        self.export_file: Optional[Path] = None
        self.auto_export_interval: Optional[float] = None
        self.last_export_time: float = time.time()
    
    def start_monitoring(self):
        """Start system monitoring."""
        with self.monitoring_lock:
            if self.is_monitoring:
                return
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        with self.monitoring_lock:
            self.is_monitoring = False
            
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
    
    def set_export_config(self, 
                         file_path: str,
                         auto_export_interval: Optional[float] = 300.0):
        """Configure automatic export.
        
        Args:
            file_path: Path to export file
            auto_export_interval: Auto export interval in seconds (None to disable)
        """
        self.export_file = Path(file_path)
        self.auto_export_interval = auto_export_interval
    
    # Event recording methods
    
    async def record_execution_start(self, execution_id: str, config: Dict[str, Any]):
        """Record SDLC execution start."""
        event = TelemetryEvent(
            timestamp=time.time(),
            event_type="execution",
            event_name="execution_start",
            data={
                "execution_id": execution_id,
                "config": config
            }
        )
        await self._add_event(event)
    
    async def record_execution_end(self, 
                                  execution_id: str, 
                                  success: bool,
                                  results: Dict[str, Any]):
        """Record SDLC execution end."""
        event = TelemetryEvent(
            timestamp=time.time(),
            event_type="execution",
            event_name="execution_end",
            data={
                "execution_id": execution_id,
                "results": results
            },
            success=success
        )
        await self._add_event(event)
    
    async def record_generation_start(self, generation: int, generation_type: str):
        """Record generation start."""
        start_time = time.time()
        self.active_generations[generation] = start_time
        
        event = TelemetryEvent(
            timestamp=start_time,
            event_type="generation",
            event_name="generation_start",
            data={
                "generation": generation,
                "generation_type": generation_type
            }
        )
        await self._add_event(event)
    
    async def record_generation_end(self, 
                                   generation: int, 
                                   success: bool,
                                   results: Dict[str, Any]):
        """Record generation end."""
        end_time = time.time()
        start_time = self.active_generations.pop(generation, end_time)
        duration = end_time - start_time
        
        # Update generation metrics
        if generation not in self.generation_metrics:
            self.generation_metrics[generation] = {
                "durations": [],
                "success_rate": [],
                "checkpoints_completed": []
            }
        
        self.generation_metrics[generation]["durations"].append(duration)
        self.generation_metrics[generation]["success_rate"].append(success)
        
        event = TelemetryEvent(
            timestamp=end_time,
            event_type="generation",
            event_name="generation_end",
            data={
                "generation": generation,
                "results": results
            },
            duration=duration,
            success=success
        )
        await self._add_event(event)
    
    async def start_checkpoint_timing(self, checkpoint: str):
        """Start timing a checkpoint."""
        start_time = time.time()
        self.active_checkpoints[checkpoint] = start_time
        
        event = TelemetryEvent(
            timestamp=start_time,
            event_type="checkpoint",
            event_name="checkpoint_start",
            data={"checkpoint": checkpoint}
        )
        await self._add_event(event)
    
    async def end_checkpoint_timing(self, 
                                   checkpoint: str, 
                                   success: bool,
                                   result: Optional[Dict[str, Any]] = None,
                                   error: Optional[str] = None):
        """End timing a checkpoint."""
        end_time = time.time()
        start_time = self.active_checkpoints.pop(checkpoint, end_time)
        duration = end_time - start_time
        
        # Update checkpoint metrics
        if checkpoint not in self.checkpoint_metrics:
            self.checkpoint_metrics[checkpoint] = []
        self.checkpoint_metrics[checkpoint].append(duration)
        
        event = TelemetryEvent(
            timestamp=end_time,
            event_type="checkpoint",
            event_name="checkpoint_end",
            data={
                "checkpoint": checkpoint,
                "result": result or {}
            },
            duration=duration,
            success=success,
            error_message=error
        )
        await self._add_event(event)
    
    async def record_quality_gate(self, 
                                 gate_name: str, 
                                 status: str,
                                 score: float,
                                 details: Dict[str, Any]):
        """Record quality gate results."""
        event = TelemetryEvent(
            timestamp=time.time(),
            event_type="quality_gate",
            event_name=gate_name,
            data={
                "status": status,
                "score": score,
                "details": details
            },
            success=(status == "passed")
        )
        await self._add_event(event)
    
    async def record_breakthrough_detection(self, 
                                          opportunities: List[Dict[str, Any]]):
        """Record breakthrough opportunity detection."""
        event = TelemetryEvent(
            timestamp=time.time(),
            event_type="research",
            event_name="breakthrough_detection",
            data={
                "opportunities_count": len(opportunities),
                "opportunities": opportunities
            }
        )
        await self._add_event(event)
    
    async def record_breakthrough_implementation(self, 
                                               opportunity_name: str,
                                               success: bool,
                                               results: Dict[str, Any]):
        """Record breakthrough implementation."""
        event = TelemetryEvent(
            timestamp=time.time(),
            event_type="research",
            event_name="breakthrough_implementation",
            data={
                "opportunity": opportunity_name,
                "results": results
            },
            success=success
        )
        await self._add_event(event)
    
    async def record_performance_metrics(self, 
                                       component: str,
                                       metrics: Dict[str, float]):
        """Record performance metrics."""
        event = TelemetryEvent(
            timestamp=time.time(),
            event_type="performance",
            event_name="metrics_recorded",
            data={
                "component": component,
                "metrics": metrics
            }
        )
        await self._add_event(event)
    
    async def collect_project_metrics(self, 
                                    project_root: str,
                                    analysis_result: Dict[str, Any]):
        """Collect project-specific metrics."""
        project_metrics = {
            "project_root": project_root,
            "project_type": analysis_result.get("project_type"),
            "language": analysis_result.get("language"),
            "framework": analysis_result.get("framework"),
            "dependencies_count": len(analysis_result.get("dependencies", [])),
            "file_count": analysis_result.get("file_structure", {}).get("total_files", 0),
            "complexity_metrics": analysis_result.get("complexity_metrics", {}),
            "quality_indicators": analysis_result.get("quality_indicators", {}),
            "research_indicators": analysis_result.get("research_indicators", {})
        }
        
        event = TelemetryEvent(
            timestamp=time.time(),
            event_type="project",
            event_name="project_metrics",
            data=project_metrics
        )
        await self._add_event(event)
    
    # Analysis and reporting methods
    
    def get_checkpoint_performance(self) -> Dict[str, Dict[str, float]]:
        """Get checkpoint performance statistics."""
        performance = {}
        
        for checkpoint, durations in self.checkpoint_metrics.items():
            if durations:
                performance[checkpoint] = {
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "execution_count": len(durations),
                    "total_time": sum(durations)
                }
        
        return performance
    
    def get_generation_performance(self) -> Dict[int, Dict[str, Any]]:
        """Get generation performance statistics."""
        performance = {}
        
        for generation, metrics in self.generation_metrics.items():
            durations = metrics.get("durations", [])
            success_rates = metrics.get("success_rate", [])
            
            if durations:
                performance[generation] = {
                    "avg_duration": sum(durations) / len(durations),
                    "success_rate": sum(success_rates) / len(success_rates) if success_rates else 0.0,
                    "execution_count": len(durations),
                    "total_time": sum(durations)
                }
        
        return performance
    
    def get_system_performance_summary(self) -> Dict[str, Any]:
        """Get system performance summary."""
        if not self.system_metrics:
            return {}
        
        cpu_values = [m.cpu_percent for m in self.system_metrics]
        memory_values = [m.memory_percent for m in self.system_metrics]
        
        return {
            "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
            "max_cpu_percent": max(cpu_values),
            "avg_memory_percent": sum(memory_values) / len(memory_values),
            "max_memory_percent": max(memory_values),
            "monitoring_duration": self.system_metrics[-1].timestamp - self.system_metrics[0].timestamp,
            "sample_count": len(self.system_metrics)
        }
    
    def get_quality_gate_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get quality gate performance summary."""
        quality_gates = {}
        
        for event in self.events:
            if event.event_type == "quality_gate":
                gate_name = event.event_name
                
                if gate_name not in quality_gates:
                    quality_gates[gate_name] = {
                        "executions": 0,
                        "passes": 0,
                        "failures": 0,
                        "avg_score": 0.0,
                        "scores": []
                    }
                
                gate_data = quality_gates[gate_name]
                gate_data["executions"] += 1
                
                if event.success:
                    gate_data["passes"] += 1
                else:
                    gate_data["failures"] += 1
                
                score = event.data.get("score", 0.0)
                gate_data["scores"].append(score)
                gate_data["avg_score"] = sum(gate_data["scores"]) / len(gate_data["scores"])
        
        return quality_gates
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get research activities summary."""
        research_summary = {
            "opportunities_detected": 0,
            "implementations_attempted": 0,
            "implementations_successful": 0,
            "domains": set(),
            "avg_confidence": 0.0
        }
        
        confidences = []
        
        for event in self.events:
            if event.event_type == "research":
                if event.event_name == "breakthrough_detection":
                    opportunities = event.data.get("opportunities", [])
                    research_summary["opportunities_detected"] += len(opportunities)
                    
                    for opp in opportunities:
                        research_summary["domains"].add(opp.get("domain", "unknown"))
                        confidences.append(opp.get("confidence", 0.0))
                
                elif event.event_name == "breakthrough_implementation":
                    research_summary["implementations_attempted"] += 1
                    if event.success:
                        research_summary["implementations_successful"] += 1
        
        if confidences:
            research_summary["avg_confidence"] = sum(confidences) / len(confidences)
        
        research_summary["domains"] = list(research_summary["domains"])
        return research_summary
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive telemetry report."""
        
        report = {
            "report_timestamp": time.time(),
            "report_datetime": datetime.now().isoformat(),
            "collection_period": {
                "start_time": self.events[0].timestamp if self.events else time.time(),
                "end_time": self.events[-1].timestamp if self.events else time.time(),
                "total_events": len(self.events)
            },
            "checkpoint_performance": self.get_checkpoint_performance(),
            "generation_performance": self.get_generation_performance(),
            "system_performance": self.get_system_performance_summary(),
            "quality_gates": self.get_quality_gate_summary(),
            "research_activities": self.get_research_summary(),
            "event_types_distribution": self._get_event_type_distribution()
        }
        
        return report
    
    def export_telemetry(self, file_path: Optional[str] = None) -> bool:
        """Export telemetry data to file."""
        
        export_path = Path(file_path) if file_path else self.export_file
        
        if not export_path:
            return False
        
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                "metadata": {
                    "export_timestamp": time.time(),
                    "export_datetime": datetime.now().isoformat(),
                    "telemetry_version": "1.0.0"
                },
                "comprehensive_report": self.generate_comprehensive_report(),
                "raw_events": [event.to_dict() for event in self.events],
                "system_metrics": [metric.to_dict() for metric in self.system_metrics]
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.last_export_time = time.time()
            return True
            
        except Exception as e:
            print(f"Failed to export telemetry: {e}")
            return False
    
    # Private methods
    
    async def _add_event(self, event: TelemetryEvent):
        """Add event to collection."""
        self.events.append(event)
        
        # Maintain max events limit
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events//2:]  # Keep recent half
        
        # Auto-export if configured
        if (self.auto_export_interval and 
            time.time() - self.last_export_time > self.auto_export_interval):
            self.export_telemetry()
    
    def _monitoring_loop(self):
        """System monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                
                # Maintain metrics limit
                if len(self.system_metrics) > self.max_events:
                    self.system_metrics = self.system_metrics[-self.max_events//2:]
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }
        
        # Process count
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            disk_usage_percent=disk.percent,
            network_io=network_io,
            process_count=process_count
        )
    
    def _get_event_type_distribution(self) -> Dict[str, int]:
        """Get distribution of event types."""
        distribution = {}
        
        for event in self.events:
            event_type = event.event_type
            distribution[event_type] = distribution.get(event_type, 0) + 1
        
        return distribution