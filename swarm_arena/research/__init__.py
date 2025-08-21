"""Research tools and algorithms for swarm intelligence."""

# Import only what we need for the breakthrough research
try:
    from .breakthrough_algorithms import CausalDiscoveryEngine, EmergentPatternDetector
    from .experiment import ExperimentLogger, ReproducibilityManager
    _imports_available = ["CausalDiscoveryEngine", "EmergentPatternDetector", "ExperimentLogger", "ReproducibilityManager"]
except ImportError:
    _imports_available = []

# Try to import other modules (graceful failure)
try:
    from .emergence import EmergenceDetector, PatternAnalyzer
    _imports_available.extend(["EmergenceDetector", "PatternAnalyzer"])
except ImportError:
    pass

try:
    from .communication import CommunicationProtocol, MessageChannel
    _imports_available.extend(["CommunicationProtocol", "MessageChannel"])
except ImportError:
    pass

try:
    from .fairness import FairnessAnalyzer
    _imports_available.append("FairnessAnalyzer")
except ImportError:
    pass

__all__ = _imports_available