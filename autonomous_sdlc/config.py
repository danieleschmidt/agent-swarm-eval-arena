"""Configuration for Autonomous SDLC system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

class ProjectType(Enum):
    """Supported project types for autonomous analysis."""
    API_SERVICE = "api_service"
    CLI_TOOL = "cli_tool"
    WEB_APPLICATION = "web_application"
    LIBRARY = "library"
    RESEARCH_PLATFORM = "research_platform"
    MACHINE_LEARNING = "machine_learning"

class GenerationStrategy(Enum):
    """Strategy for progressive enhancement."""
    SIMPLE_TO_ROBUST = "simple_to_robust"
    RESEARCH_DRIVEN = "research_driven"
    PERFORMANCE_FIRST = "performance_first"
    SECURITY_FIRST = "security_first"

@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""
    min_test_coverage: float = 0.85
    max_build_time: int = 300  # seconds
    max_security_vulnerabilities: int = 0
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "api_response_time": 200,  # ms
        "memory_usage": 512,  # MB
        "cpu_usage": 80  # %
    })
    enable_continuous_monitoring: bool = True

@dataclass
class ResearchConfig:
    """Configuration for research opportunities."""
    enable_breakthrough_detection: bool = True
    auto_implement_optimizations: bool = True
    publish_ready_documentation: bool = True
    statistical_significance_threshold: float = 0.05
    min_improvement_threshold: float = 0.10  # 10% improvement
    research_domains: List[str] = field(default_factory=lambda: [
        "algorithms", "performance", "scalability", "novel_architectures"
    ])

@dataclass
class GlobalizationConfig:
    """Configuration for global-first implementation."""
    supported_languages: List[str] = field(default_factory=lambda: [
        "en", "es", "fr", "de", "ja", "zh", "pt", "ru", "ar", "hi"
    ])
    compliance_frameworks: List[str] = field(default_factory=lambda: [
        "GDPR", "CCPA", "PDPA", "LGPD", "SOC2", "ISO27001"
    ])
    deployment_regions: List[str] = field(default_factory=lambda: [
        "us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "ap-northeast-1"
    ])

@dataclass
class AutonomousSDLCConfig:
    """Master configuration for Autonomous SDLC."""
    
    # Project identification
    project_type: Optional[ProjectType] = None
    project_language: str = "python"
    project_domain: str = "general"
    
    # Generation strategy
    generation_strategy: GenerationStrategy = GenerationStrategy.SIMPLE_TO_ROBUST
    max_generations: int = 3
    auto_proceed_generations: bool = True
    
    # Quality gates
    quality_gates: QualityGateConfig = field(default_factory=QualityGateConfig)
    
    # Research capabilities
    research_config: ResearchConfig = field(default_factory=ResearchConfig)
    
    # Globalization
    globalization: GlobalizationConfig = field(default_factory=GlobalizationConfig)
    
    # Execution settings
    autonomous_execution: bool = True
    parallel_execution: bool = True
    max_concurrent_tasks: int = 8
    
    # Monitoring and telemetry
    enable_telemetry: bool = True
    telemetry_endpoint: Optional[str] = None
    real_time_monitoring: bool = True
    
    # Security
    security_scanning: bool = True
    dependency_scanning: bool = True
    secrets_scanning: bool = True
    
    # Performance optimization
    enable_auto_scaling: bool = True
    performance_profiling: bool = True
    cache_optimization: bool = True
    
    # Documentation
    auto_generate_docs: bool = True
    api_documentation: bool = True
    research_documentation: bool = True
    
    # Deployment
    auto_deployment: bool = False  # Conservative default
    deployment_environments: List[str] = field(default_factory=lambda: ["staging"])
    infrastructure_as_code: bool = True
    
    # Custom configurations
    custom_checkpoints: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    
    def get_checkpoints_for_project_type(self) -> List[str]:
        """Get appropriate checkpoints based on project type."""
        checkpoint_map = {
            ProjectType.API_SERVICE: [
                "foundation", "data_layer", "authentication", 
                "endpoints", "testing", "monitoring", "deployment"
            ],
            ProjectType.CLI_TOOL: [
                "structure", "commands", "configuration", 
                "plugins", "testing", "documentation"
            ],
            ProjectType.WEB_APPLICATION: [
                "frontend", "backend", "state_management", 
                "ui_components", "testing", "deployment"
            ],
            ProjectType.LIBRARY: [
                "core_modules", "public_api", "examples", 
                "documentation", "testing", "packaging"
            ],
            ProjectType.RESEARCH_PLATFORM: [
                "research_framework", "breakthrough_algorithms", "experimental_validation",
                "publication_preparation", "benchmarking", "peer_review_ready"
            ],
            ProjectType.MACHINE_LEARNING: [
                "data_pipeline", "model_architecture", "training", 
                "evaluation", "deployment", "monitoring"
            ]
        }
        
        if self.project_type:
            return checkpoint_map.get(self.project_type, checkpoint_map[ProjectType.API_SERVICE])
        return checkpoint_map[ProjectType.API_SERVICE]
    
    def is_research_enabled(self) -> bool:
        """Check if research mode is enabled."""
        return self.research_config.enable_breakthrough_detection
    
    def get_quality_thresholds(self) -> Dict[str, Any]:
        """Get quality gate thresholds."""
        return {
            "test_coverage": self.quality_gates.min_test_coverage,
            "build_time": self.quality_gates.max_build_time,
            "security": self.quality_gates.max_security_vulnerabilities,
            "performance": self.quality_gates.performance_thresholds
        }