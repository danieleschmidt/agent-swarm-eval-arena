"""Experiment management and reproducibility tools."""

import json
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
import numpy as np
import pickle
import subprocess
import sys


@dataclass
class ExperimentConfig:
    """Configuration for reproducible experiments."""
    
    name: str
    description: str
    seed: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    git_commit: Optional[str] = None
    python_version: str = field(default_factory=lambda: sys.version)
    dependencies: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Automatically capture environment information."""
        if not self.environment:
            self.environment = self._capture_environment()
        
        if not self.git_commit:
            self.git_commit = self._get_git_commit()
    
    def _capture_environment(self) -> Dict[str, str]:
        """Capture relevant environment variables."""
        relevant_vars = [
            'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'PYTHONPATH', 'PATH'
        ]
        
        env = {}
        for var in relevant_vars:
            value = os.environ.get(var)
            if value is not None:
                env[var] = value
        
        return env
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    
    config: ExperimentConfig
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # filename -> path
    execution_time: float = 0.0
    memory_usage: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def save(self, base_path: Union[str, Path]) -> None:
        """Save experiment result."""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = base_path / "config.json"
        self.config.save(config_path)
        
        # Save metrics
        metrics_path = base_path / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save result summary
        result_path = base_path / "result.json"
        result_data = {
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "success": self.success,
            "error_message": self.error_message,
            "artifacts": self.artifacts
        }
        
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)


class ExperimentLogger:
    """Logger for tracking experiments and ensuring reproducibility."""
    
    def __init__(self, 
                 base_path: Union[str, Path] = "experiments",
                 auto_save: bool = True):
        """Initialize experiment logger.
        
        Args:
            base_path: Base directory for storing experiments
            auto_save: Whether to automatically save results
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.auto_save = auto_save
        self.current_experiment: Optional[ExperimentResult] = None
        self.start_time: Optional[float] = None
        
    def start_experiment(self, 
                        name: str,
                        description: str = "",
                        parameters: Optional[Dict[str, Any]] = None,
                        seed: Optional[int] = None) -> ExperimentResult:
        """Start a new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            parameters: Experiment parameters
            seed: Random seed
            
        Returns:
            ExperimentResult object
        """
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        
        if parameters is None:
            parameters = {}
        
        # Create experiment configuration
        config = ExperimentConfig(
            name=name,
            description=description,
            seed=seed,
            parameters=parameters
        )
        
        # Create experiment result
        self.current_experiment = ExperimentResult(config=config)
        self.start_time = time.time()
        
        print(f"🧪 Starting experiment: {name}")
        print(f"📝 Description: {description}")
        print(f"🎲 Seed: {seed}")
        
        return self.current_experiment
    
    def log_metric(self, name: str, value: Any) -> None:
        """Log a metric for the current experiment.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if self.current_experiment is None:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment.metrics[name] = value
        print(f"📊 Logged metric {name}: {value}")
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log multiple metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        for name, value in metrics.items():
            self.log_metric(name, value)
    
    def log_artifact(self, name: str, filepath: Union[str, Path]) -> None:
        """Log an artifact file.
        
        Args:
            name: Artifact name
            filepath: Path to artifact file
        """
        if self.current_experiment is None:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Artifact file not found: {filepath}")
        
        self.current_experiment.artifacts[name] = str(filepath.absolute())
        print(f"📎 Logged artifact {name}: {filepath}")
    
    def finish_experiment(self, 
                         success: bool = True,
                         error_message: Optional[str] = None) -> Optional[Path]:
        """Finish the current experiment.
        
        Args:
            success: Whether the experiment succeeded
            error_message: Error message if failed
            
        Returns:
            Path to saved experiment directory if auto_save is True
        """
        if self.current_experiment is None:
            raise RuntimeError("No active experiment to finish.")
        
        # Calculate execution time
        if self.start_time is not None:
            self.current_experiment.execution_time = time.time() - self.start_time
        
        # Set success status
        self.current_experiment.success = success
        self.current_experiment.error_message = error_message
        
        print(f"🏁 Experiment finished: {self.current_experiment.config.name}")
        print(f"⏱️  Execution time: {self.current_experiment.execution_time:.2f}s")
        print(f"✅ Success: {success}")
        
        if error_message:
            print(f"❌ Error: {error_message}")
        
        # Save experiment if auto_save is enabled
        saved_path = None
        if self.auto_save:
            saved_path = self.save_experiment()
        
        # Reset current experiment
        self.current_experiment = None
        self.start_time = None
        
        return saved_path
    
    def save_experiment(self, 
                       custom_path: Optional[Union[str, Path]] = None) -> Path:
        """Save the current experiment.
        
        Args:
            custom_path: Custom path for saving (optional)
            
        Returns:
            Path where experiment was saved
        """
        if self.current_experiment is None:
            raise RuntimeError("No active experiment to save.")
        
        if custom_path is None:
            # Generate unique experiment directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            exp_name = self.current_experiment.config.name.replace(" ", "_")
            exp_dir = self.base_path / f"{timestamp}_{exp_name}"
        else:
            exp_dir = Path(custom_path)
        
        # Save experiment
        self.current_experiment.save(exp_dir)
        
        print(f"💾 Experiment saved to: {exp_dir}")
        return exp_dir
    
    def load_experiment(self, experiment_path: Union[str, Path]) -> ExperimentResult:
        """Load a saved experiment.
        
        Args:
            experiment_path: Path to experiment directory
            
        Returns:
            Loaded ExperimentResult
        """
        exp_path = Path(experiment_path)
        
        # Load configuration
        config_path = exp_path / "config.json"
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = ExperimentConfig(**config_data)
        
        # Load metrics
        metrics_path = exp_path / "metrics.json"
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Load result data
        result_path = exp_path / "result.json"
        with open(result_path, 'r') as f:
            result_data = json.load(f)
        
        # Create ExperimentResult
        result = ExperimentResult(
            config=config,
            metrics=metrics,
            artifacts=result_data.get("artifacts", {}),
            execution_time=result_data.get("execution_time", 0.0),
            memory_usage=result_data.get("memory_usage"),
            success=result_data.get("success", True),
            error_message=result_data.get("error_message")
        )
        
        return result
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments in the base directory.
        
        Returns:
            List of experiment summaries
        """
        experiments = []
        
        for exp_dir in self.base_path.iterdir():
            if exp_dir.is_dir():
                try:
                    config_path = exp_dir / "config.json"
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config_data = json.load(f)
                        
                        result_path = exp_dir / "result.json"
                        result_data = {}
                        if result_path.exists():
                            with open(result_path, 'r') as f:
                                result_data = json.load(f)
                        
                        experiments.append({
                            "path": str(exp_dir),
                            "name": config_data.get("name", "Unknown"),
                            "timestamp": config_data.get("timestamp", 0),
                            "success": result_data.get("success", False),
                            "execution_time": result_data.get("execution_time", 0)
                        })
                
                except Exception as e:
                    print(f"Warning: Could not load experiment from {exp_dir}: {e}")
        
        # Sort by timestamp (newest first)
        experiments.sort(key=lambda x: x["timestamp"], reverse=True)
        return experiments


class ReproducibilityManager:
    """Manager for ensuring experiment reproducibility."""
    
    def __init__(self):
        self.checkpoints: Dict[str, Any] = {}
    
    def set_seeds(self, seed: int) -> None:
        """Set all random seeds for reproducibility.
        
        Args:
            seed: Random seed value
        """
        np.random.seed(seed)
        
        # Try to set other common library seeds
        try:
            import random
            random.seed(seed)
        except ImportError:
            pass
        
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
        
        print(f"🎲 Set random seeds to {seed}")
    
    def checkpoint_state(self, name: str, state: Any) -> None:
        """Save a checkpoint of the current state.
        
        Args:
            name: Checkpoint name
            state: State to checkpoint
        """
        # Create deep copy to avoid reference issues
        if hasattr(state, 'copy'):
            checkpointed_state = state.copy()
        else:
            try:
                checkpointed_state = pickle.loads(pickle.dumps(state))
            except:
                # Fallback for non-picklable objects
                checkpointed_state = str(state)
        
        self.checkpoints[name] = checkpointed_state
        print(f"💾 Checkpointed state: {name}")
    
    def restore_state(self, name: str) -> Any:
        """Restore a checkpointed state.
        
        Args:
            name: Checkpoint name
            
        Returns:
            Restored state
        """
        if name not in self.checkpoints:
            raise KeyError(f"Checkpoint '{name}' not found")
        
        state = self.checkpoints[name]
        print(f"🔄 Restored state: {name}")
        return state
    
    def verify_reproducibility(self, 
                             experiment_func: callable,
                             config: ExperimentConfig,
                             num_runs: int = 3) -> Dict[str, Any]:
        """Verify that an experiment is reproducible.
        
        Args:
            experiment_func: Function that runs the experiment
            config: Experiment configuration
            num_runs: Number of verification runs
            
        Returns:
            Reproducibility verification results
        """
        print(f"🔍 Verifying reproducibility with {num_runs} runs...")
        
        results = []
        checksums = []
        
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            
            # Set seed for this run
            self.set_seeds(config.seed)
            
            # Run experiment
            try:
                result = experiment_func(config)
                results.append(result)
                
                # Calculate checksum of result
                result_str = json.dumps(result, sort_keys=True, default=str)
                checksum = hashlib.md5(result_str.encode()).hexdigest()
                checksums.append(checksum)
                
            except Exception as e:
                print(f"❌ Run {run + 1} failed: {e}")
                return {
                    "reproducible": False,
                    "error": str(e),
                    "successful_runs": run
                }
        
        # Check if all runs produced identical results
        all_identical = len(set(checksums)) == 1
        
        verification_result = {
            "reproducible": all_identical,
            "num_runs": num_runs,
            "successful_runs": len(results),
            "checksums": checksums,
            "results": results
        }
        
        if all_identical:
            print("✅ Experiment is reproducible!")
        else:
            print("❌ Experiment is NOT reproducible!")
            print("   Different runs produced different results.")
        
        return verification_result
    
    def generate_reproducibility_report(self, config: ExperimentConfig) -> str:
        """Generate a reproducibility report.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Formatted report string
        """
        report = f"""
🔬 REPRODUCIBILITY REPORT
═══════════════════════════

📋 EXPERIMENT: {config.name}
📝 Description: {config.description}

🔧 CONFIGURATION:
  • Seed: {config.seed}
  • Git Commit: {config.git_commit or 'Unknown'}
  • Python Version: {config.python_version}
  • Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(config.timestamp))}

🌍 ENVIRONMENT:
"""
        
        for key, value in config.environment.items():
            report += f"  • {key}: {value}\n"
        
        report += f"""
📦 PARAMETERS:
"""
        
        for key, value in config.parameters.items():
            report += f"  • {key}: {value}\n"
        
        report += f"""
🔍 VERIFICATION:
  To reproduce this experiment:
  1. Checkout git commit: {config.git_commit or 'N/A'}
  2. Set environment variables as listed above
  3. Use seed: {config.seed}
  4. Run with parameters as listed above

✅ This configuration should produce identical results.
"""
        
        return report