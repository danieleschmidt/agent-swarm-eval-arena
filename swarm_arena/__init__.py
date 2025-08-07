"""
Swarm Arena: Real-Time Arena for Multi-Agent Reinforcement Learning

A massively scalable evaluation platform for Multi-Agent Reinforcement Learning (MARL) 
research supporting 1000+ concurrent agents with real-time telemetry and comprehensive 
benchmarking capabilities.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

from .core.arena import Arena
from .core.agent import (
    Agent, BaseAgent, CooperativeAgent, CompetitiveAgent, RandomAgent,
    LearningAgent, HierarchicalAgent, SwarmAgent, AdaptiveAgent
)
from .core.sentiment_aware_agent import (
    SentimentAwareAgent, EmotionalCooperativeAgent, EmotionalCompetitiveAgent,
    EmotionalAdaptiveAgent, SentimentAwareAgentConfig
)
from .core.config import SwarmConfig
from .core.environment import Environment
from .utils.seeding import set_global_seed
from .monitoring.telemetry import TelemetryCollector, TelemetryData
from .monitoring.sentiment_telemetry import SentimentTelemetryCollector, SentimentMetrics
from .sentiment.processor import SentimentProcessor, SentimentData
from .sentiment.emotional_state import EmotionalState, EmotionType
from .sentiment.contagion import SentimentContagion, ContagionParameters
from .sentiment.memory import SentimentMemoryBuffer

# Main exports for public API
__all__ = [
    # Core components
    "Arena",
    "Agent", 
    "BaseAgent",
    "CooperativeAgent",
    "CompetitiveAgent", 
    "RandomAgent",
    "LearningAgent",
    "HierarchicalAgent",
    "SwarmAgent",
    "AdaptiveAgent",
    "SwarmConfig",
    "Environment",
    "set_global_seed",
    
    # Sentiment-aware agents
    "SentimentAwareAgent",
    "EmotionalCooperativeAgent", 
    "EmotionalCompetitiveAgent",
    "EmotionalAdaptiveAgent",
    "SentimentAwareAgentConfig",
    
    # Sentiment processing
    "SentimentProcessor",
    "SentimentData",
    "EmotionalState", 
    "EmotionType",
    "SentimentContagion",
    "ContagionParameters",
    "SentimentMemoryBuffer",
    
    # Monitoring and telemetry
    "TelemetryCollector",
    "TelemetryData",
    "SentimentTelemetryCollector",
    "SentimentMetrics",
]