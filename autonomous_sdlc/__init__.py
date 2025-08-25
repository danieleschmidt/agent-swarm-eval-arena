"""
Autonomous SDLC: Self-executing software development lifecycle
with progressive enhancement and breakthrough research capabilities.
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs"

from .core.execution_engine import AutonomousExecutionEngine
from .core.generation_manager import GenerationManager, Generation
from .core.quality_gates import QualityGateManager
from .research.breakthrough_detector import BreakthroughOpportunityDetector
from .monitoring.sdlc_telemetry import SDLCTelemetryCollector
from .config import AutonomousSDLCConfig

__all__ = [
    "AutonomousExecutionEngine",
    "GenerationManager", 
    "Generation",
    "QualityGateManager",
    "BreakthroughOpportunityDetector",
    "SDLCTelemetryCollector",
    "AutonomousSDLCConfig",
]