"""Research components for autonomous breakthrough detection and implementation."""

from .breakthrough_detector import BreakthroughOpportunityDetector
from .research_validator import ResearchValidator
from .publication_engine import PublicationEngine

__all__ = [
    "BreakthroughOpportunityDetector",
    "ResearchValidator", 
    "PublicationEngine",
]