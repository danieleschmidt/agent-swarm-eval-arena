"""Auto-scaling module for Swarm Arena."""

from .auto_scaler import (
    AutoScaler, ScalingEngine, ScalingPolicy, ScalingMetrics, ScalingDecision,
    ScalingDirection, MetricsCollector, auto_scaler, DEFAULT_POLICIES
)

__all__ = [
    "AutoScaler",
    "ScalingEngine",
    "ScalingPolicy", 
    "ScalingMetrics",
    "ScalingDecision",
    "ScalingDirection",
    "MetricsCollector",
    "auto_scaler",
    "DEFAULT_POLICIES",
]