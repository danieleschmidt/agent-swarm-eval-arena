"""
Checkpoint Manager: Manages execution checkpoints, dependencies,
and rollback capabilities for autonomous SDLC execution.
"""

import json
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import hashlib

class CheckpointStatus(Enum):
    """Status of checkpoint execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"

@dataclass
class CheckpointState:
    """State information for a checkpoint."""
    name: str
    status: CheckpointStatus
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    generation: int = 1
    dependencies: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)  # Files created/modified
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    rollback_data: Optional[Dict[str, Any]] = None
    
    def duration(self) -> float:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "generation": self.generation,
            "dependencies": self.dependencies,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
            "error_message": self.error_message,
            "duration": self.duration()
        }

class CheckpointManager:
    """Manages checkpoint execution state and dependencies."""
    
    def __init__(self):
        self.checkpoints: Dict[str, CheckpointState] = {}
        self.execution_order: List[str] = []
        self.state_file: Optional[Path] = None
        self.backup_states: List[Dict[str, Any]] = []
        
        # Dependency graph
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependencies: Dict[str, Set[str]] = {}
    
    def initialize_checkpoint(self, 
                            name: str, 
                            dependencies: List[str] = None,
                            generation: int = 1) -> CheckpointState:
        """Initialize a new checkpoint."""
        
        checkpoint = CheckpointState(
            name=name,
            status=CheckpointStatus.PENDING,
            generation=generation,
            dependencies=dependencies or []
        )
        
        self.checkpoints[name] = checkpoint
        
        # Update dependency graph
        self._update_dependency_graph(name, dependencies or [])
        
        return checkpoint
    
    def start_checkpoint(self, name: str) -> CheckpointState:
        """Mark checkpoint as started."""
        
        if name not in self.checkpoints:
            raise ValueError(f"Checkpoint {name} not initialized")
        
        checkpoint = self.checkpoints[name]
        
        # Check dependencies
        if not self._are_dependencies_satisfied(name):
            unsatisfied = self._get_unsatisfied_dependencies(name)
            raise RuntimeError(f"Dependencies not satisfied for {name}: {unsatisfied}")
        
        checkpoint.status = CheckpointStatus.RUNNING
        checkpoint.start_time = time.time()
        
        # Add to execution order
        if name not in self.execution_order:
            self.execution_order.append(name)
        
        self._save_state()
        return checkpoint
    
    def complete_checkpoint(self, 
                          name: str, 
                          artifacts: List[str] = None,
                          metadata: Dict[str, Any] = None) -> CheckpointState:
        """Mark checkpoint as completed."""
        
        if name not in self.checkpoints:
            raise ValueError(f"Checkpoint {name} not found")
        
        checkpoint = self.checkpoints[name]
        checkpoint.status = CheckpointStatus.COMPLETED
        checkpoint.end_time = time.time()
        
        if artifacts:
            checkpoint.artifacts.extend(artifacts)
        
        if metadata:
            checkpoint.metadata.update(metadata)
        
        self._save_state()
        return checkpoint
    
    def fail_checkpoint(self, name: str, error_message: str) -> CheckpointState:
        """Mark checkpoint as failed."""
        
        if name not in self.checkpoints:
            raise ValueError(f"Checkpoint {name} not found")
        
        checkpoint = self.checkpoints[name]
        checkpoint.status = CheckpointStatus.FAILED
        checkpoint.end_time = time.time()
        checkpoint.error_message = error_message
        
        self._save_state()
        return checkpoint
    
    def skip_checkpoint(self, name: str, reason: str) -> CheckpointState:
        """Mark checkpoint as skipped."""
        
        if name not in self.checkpoints:
            raise ValueError(f"Checkpoint {name} not found")
        
        checkpoint = self.checkpoints[name]
        checkpoint.status = CheckpointStatus.SKIPPED
        checkpoint.metadata["skip_reason"] = reason
        
        self._save_state()
        return checkpoint
    
    def rollback_checkpoint(self, name: str) -> CheckpointState:
        """Rollback a completed checkpoint."""
        
        if name not in self.checkpoints:
            raise ValueError(f"Checkpoint {name} not found")
        
        checkpoint = self.checkpoints[name]
        
        if checkpoint.status != CheckpointStatus.COMPLETED:
            raise RuntimeError(f"Cannot rollback checkpoint {name} with status {checkpoint.status}")
        
        # Rollback dependent checkpoints first
        dependents = self._get_dependent_checkpoints(name)
        for dependent in dependents:
            if self.checkpoints[dependent].status == CheckpointStatus.COMPLETED:
                self.rollback_checkpoint(dependent)
        
        # Perform rollback
        checkpoint.status = CheckpointStatus.ROLLED_BACK
        checkpoint.rollback_data = {
            "rollback_time": time.time(),
            "original_completion_time": checkpoint.end_time
        }
        
        self._save_state()
        return checkpoint
    
    def get_execution_plan(self, checkpoints: List[str]) -> List[str]:
        """Get optimized execution plan for checkpoints."""
        
        # Topological sort considering dependencies
        plan = []
        visited = set()
        temp_visited = set()
        
        def visit(checkpoint_name: str):
            if checkpoint_name in temp_visited:
                raise RuntimeError(f"Circular dependency detected involving {checkpoint_name}")
            
            if checkpoint_name in visited:
                return
            
            temp_visited.add(checkpoint_name)
            
            # Visit dependencies first
            for dep in self.dependency_graph.get(checkpoint_name, set()):
                if dep in checkpoints:  # Only consider requested checkpoints
                    visit(dep)
            
            temp_visited.remove(checkpoint_name)
            visited.add(checkpoint_name)
            plan.append(checkpoint_name)
        
        # Visit all requested checkpoints
        for checkpoint in checkpoints:
            if checkpoint not in visited:
                visit(checkpoint)
        
        return plan
    
    def get_parallel_execution_groups(self, checkpoints: List[str]) -> List[List[str]]:
        """Group checkpoints that can be executed in parallel."""
        
        plan = self.get_execution_plan(checkpoints)
        groups = []
        remaining = set(plan)
        
        while remaining:
            # Find checkpoints with no remaining dependencies
            ready = []
            for checkpoint in remaining:
                deps = self.dependency_graph.get(checkpoint, set())
                if not (deps & remaining):  # No dependencies remaining
                    ready.append(checkpoint)
            
            if not ready:
                # Should not happen with valid dependency graph
                raise RuntimeError("Circular dependency or invalid state")
            
            groups.append(ready)
            remaining -= set(ready)
        
        return groups
    
    def get_checkpoint_status(self, name: str) -> Optional[CheckpointStatus]:
        """Get status of a checkpoint."""
        checkpoint = self.checkpoints.get(name)
        return checkpoint.status if checkpoint else None
    
    def get_completed_checkpoints(self) -> List[str]:
        """Get list of completed checkpoints."""
        return [
            name for name, checkpoint in self.checkpoints.items()
            if checkpoint.status == CheckpointStatus.COMPLETED
        ]
    
    def get_failed_checkpoints(self) -> List[str]:
        """Get list of failed checkpoints."""
        return [
            name for name, checkpoint in self.checkpoints.items()
            if checkpoint.status == CheckpointStatus.FAILED
        ]
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get summary of all checkpoints."""
        
        summary = {
            "total_checkpoints": len(self.checkpoints),
            "completed": 0,
            "failed": 0,
            "running": 0,
            "pending": 0,
            "skipped": 0,
            "total_execution_time": 0.0
        }
        
        for checkpoint in self.checkpoints.values():
            summary[checkpoint.status.value] += 1
            summary["total_execution_time"] += checkpoint.duration()
        
        summary["completion_rate"] = (
            summary["completed"] / summary["total_checkpoints"] 
            if summary["total_checkpoints"] > 0 else 0.0
        )
        
        return summary
    
    def export_checkpoint_report(self) -> Dict[str, Any]:
        """Export detailed checkpoint report."""
        
        return {
            "summary": self.get_checkpoint_summary(),
            "checkpoints": {
                name: checkpoint.to_dict() 
                for name, checkpoint in self.checkpoints.items()
            },
            "execution_order": self.execution_order,
            "dependency_graph": {
                name: list(deps) 
                for name, deps in self.dependency_graph.items()
            },
            "timestamp": time.time()
        }
    
    def set_state_file(self, file_path: str):
        """Set file path for persistent state storage."""
        self.state_file = Path(file_path)
    
    def load_state(self, file_path: Optional[str] = None) -> bool:
        """Load checkpoint state from file."""
        
        state_path = Path(file_path) if file_path else self.state_file
        
        if not state_path or not state_path.exists():
            return False
        
        try:
            with open(state_path, 'r') as f:
                data = json.load(f)
            
            # Restore checkpoints
            for name, checkpoint_data in data.get("checkpoints", {}).items():
                checkpoint = CheckpointState(
                    name=checkpoint_data["name"],
                    status=CheckpointStatus(checkpoint_data["status"]),
                    start_time=checkpoint_data.get("start_time"),
                    end_time=checkpoint_data.get("end_time"),
                    generation=checkpoint_data.get("generation", 1),
                    dependencies=checkpoint_data.get("dependencies", []),
                    artifacts=checkpoint_data.get("artifacts", []),
                    metadata=checkpoint_data.get("metadata", {}),
                    error_message=checkpoint_data.get("error_message")
                )
                self.checkpoints[name] = checkpoint
            
            # Restore execution order
            self.execution_order = data.get("execution_order", [])
            
            # Restore dependency graph
            for name, deps in data.get("dependency_graph", {}).items():
                self.dependency_graph[name] = set(deps)
            
            self._rebuild_reverse_dependencies()
            return True
            
        except Exception as e:
            print(f"Failed to load checkpoint state: {e}")
            return False
    
    def create_backup(self) -> str:
        """Create a backup of current state."""
        
        backup_data = self.export_checkpoint_report()
        backup_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        self.backup_states.append({
            "id": backup_id,
            "timestamp": time.time(),
            "data": backup_data
        })
        
        # Keep only last 10 backups
        if len(self.backup_states) > 10:
            self.backup_states = self.backup_states[-10:]
        
        return backup_id
    
    def restore_backup(self, backup_id: str) -> bool:
        """Restore from a backup."""
        
        backup = next(
            (b for b in self.backup_states if b["id"] == backup_id), 
            None
        )
        
        if not backup:
            return False
        
        try:
            data = backup["data"]
            
            # Clear current state
            self.checkpoints.clear()
            self.execution_order.clear()
            self.dependency_graph.clear()
            
            # Restore from backup
            for name, checkpoint_data in data.get("checkpoints", {}).items():
                checkpoint = CheckpointState(
                    name=checkpoint_data["name"],
                    status=CheckpointStatus(checkpoint_data["status"]),
                    start_time=checkpoint_data.get("start_time"),
                    end_time=checkpoint_data.get("end_time"),
                    generation=checkpoint_data.get("generation", 1),
                    dependencies=checkpoint_data.get("dependencies", []),
                    artifacts=checkpoint_data.get("artifacts", []),
                    metadata=checkpoint_data.get("metadata", {}),
                    error_message=checkpoint_data.get("error_message")
                )
                self.checkpoints[name] = checkpoint
            
            self.execution_order = data.get("execution_order", [])
            
            for name, deps in data.get("dependency_graph", {}).items():
                self.dependency_graph[name] = set(deps)
            
            self._rebuild_reverse_dependencies()
            self._save_state()
            
            return True
            
        except Exception as e:
            print(f"Failed to restore backup: {e}")
            return False
    
    # Private methods
    
    def _update_dependency_graph(self, checkpoint: str, dependencies: List[str]):
        """Update dependency graph for a checkpoint."""
        
        self.dependency_graph[checkpoint] = set(dependencies)
        
        # Update reverse dependencies
        for dep in dependencies:
            if dep not in self.reverse_dependencies:
                self.reverse_dependencies[dep] = set()
            self.reverse_dependencies[dep].add(checkpoint)
    
    def _rebuild_reverse_dependencies(self):
        """Rebuild reverse dependency graph."""
        
        self.reverse_dependencies.clear()
        
        for checkpoint, deps in self.dependency_graph.items():
            for dep in deps:
                if dep not in self.reverse_dependencies:
                    self.reverse_dependencies[dep] = set()
                self.reverse_dependencies[dep].add(checkpoint)
    
    def _are_dependencies_satisfied(self, checkpoint: str) -> bool:
        """Check if all dependencies for a checkpoint are satisfied."""
        
        dependencies = self.dependency_graph.get(checkpoint, set())
        
        for dep in dependencies:
            dep_checkpoint = self.checkpoints.get(dep)
            if not dep_checkpoint or dep_checkpoint.status != CheckpointStatus.COMPLETED:
                return False
        
        return True
    
    def _get_unsatisfied_dependencies(self, checkpoint: str) -> List[str]:
        """Get list of unsatisfied dependencies."""
        
        dependencies = self.dependency_graph.get(checkpoint, set())
        unsatisfied = []
        
        for dep in dependencies:
            dep_checkpoint = self.checkpoints.get(dep)
            if not dep_checkpoint or dep_checkpoint.status != CheckpointStatus.COMPLETED:
                unsatisfied.append(dep)
        
        return unsatisfied
    
    def _get_dependent_checkpoints(self, checkpoint: str) -> List[str]:
        """Get checkpoints that depend on the given checkpoint."""
        return list(self.reverse_dependencies.get(checkpoint, set()))
    
    def _save_state(self):
        """Save current state to file."""
        
        if not self.state_file:
            return
        
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.state_file, 'w') as f:
                json.dump(self.export_checkpoint_report(), f, indent=2)
                
        except Exception as e:
            print(f"Failed to save checkpoint state: {e}")