"""
Sentiment analysis module for multi-agent systems.

This module provides sentiment analysis capabilities for agents in the swarm arena,
enabling emotion-aware decision making and behavioral adaptation.
"""

from .processor import SentimentProcessor, SentimentData
from .emotional_state import EmotionalState, EmotionType
from .contagion import SentimentContagion
from .memory import SentimentMemoryBuffer

__all__ = [
    "SentimentProcessor",
    "SentimentData", 
    "EmotionalState",
    "EmotionType",
    "SentimentContagion",
    "SentimentMemoryBuffer"
]