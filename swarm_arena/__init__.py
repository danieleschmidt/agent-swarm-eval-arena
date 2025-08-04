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
from .core.agent import Agent, BaseAgent
from .core.config import SwarmConfig
from .core.environment import Environment
from .utils.seeding import set_global_seed

# Main exports for public API
__all__ = [
    "Arena",
    "Agent", 
    "BaseAgent",
    "SwarmConfig",
    "Environment",
    "set_global_seed",
]