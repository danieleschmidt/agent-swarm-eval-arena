"""Production deployment configuration and infrastructure."""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration for production."""
    host: str = "localhost"
    port: int = 5432
    database: str = "swarm_arena"
    username: str = "swarm_user"
    password: str = ""
    ssl_mode: str = "require"
    pool_size: int = 20
    max_overflow: int = 30
    
    def get_connection_string(self) -> str:
        """Get database connection string."""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )


@dataclass
class RedisConfig:
    """Redis configuration for caching and sessions."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    database: int = 0
    ssl: bool = False
    pool_size: int = 10
    
    def get_connection_string(self) -> str:
        """Get Redis connection string."""
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.database}"


@dataclass
class SecurityConfig:
    """Security configuration for production."""
    secret_key: str = ""
    jwt_secret: str = ""
    bcrypt_rounds: int = 12
    session_timeout: int = 3600  # 1 hour
    rate_limit_per_minute: int = 60
    cors_origins: list = field(default_factory=lambda: ["https://your-domain.com"])
    csrf_enabled: bool = True
    
    def __post_init__(self):
        """Generate secrets if not provided."""
        if not self.secret_key:
            import secrets
            self.secret_key = secrets.token_hex(32)
        
        if not self.jwt_secret:
            import secrets
            self.jwt_secret = secrets.token_hex(32)


@dataclass
class LoggingConfig:
    """Logging configuration for production."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "/var/log/swarm_arena/app.log"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 10
    
    # Structured logging
    json_format: bool = True
    include_trace_id: bool = True
    
    # Log levels by component
    component_levels: Dict[str, str] = field(default_factory=lambda: {
        "swarm_arena.core": "INFO",
        "swarm_arena.security": "WARNING",
        "swarm_arena.optimization": "DEBUG",
        "swarm_arena.monitoring": "INFO"
    })


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    metrics_enabled: bool = True
    metrics_port: int = 9090
    health_check_port: int = 8080
    tracing_enabled: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    
    # Alert thresholds
    cpu_alert_threshold: float = 80.0
    memory_alert_threshold: float = 85.0
    error_rate_threshold: float = 5.0  # 5% error rate
    response_time_threshold: float = 2000  # 2 seconds
    
    # Prometheus configuration
    prometheus_enabled: bool = True
    prometheus_scrape_interval: int = 15  # seconds


@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    cooldown_period: int = 300  # 5 minutes


@dataclass
class ProductionConfig:
    """Complete production configuration."""
    
    # Environment
    environment: str = "production"
    debug: bool = False
    testing: bool = False
    
    # Service configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    worker_class: str = "uvicorn.workers.UvicornWorker"
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    
    # Resource limits
    max_agents_per_simulation: int = 10000
    max_concurrent_simulations: int = 100
    max_simulation_duration: int = 3600  # 1 hour
    max_memory_per_worker: str = "2Gi"
    max_cpu_per_worker: str = "1000m"
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "distributed_computing": True,
        "auto_scaling": True,
        "advanced_analytics": True,
        "real_time_monitoring": True,
        "gpu_acceleration": False,
        "experimental_features": False
    })
    
    @classmethod
    def from_environment(cls) -> 'ProductionConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Database configuration
        config.database.host = os.getenv("DB_HOST", config.database.host)
        config.database.port = int(os.getenv("DB_PORT", config.database.port))
        config.database.database = os.getenv("DB_NAME", config.database.database)
        config.database.username = os.getenv("DB_USER", config.database.username)
        config.database.password = os.getenv("DB_PASSWORD", config.database.password)
        
        # Redis configuration
        config.redis.host = os.getenv("REDIS_HOST", config.redis.host)
        config.redis.port = int(os.getenv("REDIS_PORT", config.redis.port))
        config.redis.password = os.getenv("REDIS_PASSWORD", config.redis.password)
        
        # Security configuration
        config.security.secret_key = os.getenv("SECRET_KEY", config.security.secret_key)
        config.security.jwt_secret = os.getenv("JWT_SECRET", config.security.jwt_secret)
        
        # Service configuration
        config.host = os.getenv("HOST", config.host)
        config.port = int(os.getenv("PORT", config.port))
        config.workers = int(os.getenv("WORKERS", config.workers))
        
        # Monitoring configuration
        config.monitoring.jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", config.monitoring.jaeger_endpoint)
        
        return config
    
    def validate(self) -> list[str]:
        """Validate production configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check required secrets
        if not self.security.secret_key:
            errors.append("SECRET_KEY is required in production")
        
        if not self.security.jwt_secret:
            errors.append("JWT_SECRET is required in production")
        
        if not self.database.password:
            errors.append("Database password is required in production")
        
        # Check security settings
        if self.debug:
            errors.append("Debug mode should be disabled in production")
        
        if self.security.bcrypt_rounds < 10:
            errors.append("BCrypt rounds should be at least 10 in production")
        
        # Check resource limits
        if self.max_agents_per_simulation > 50000:
            errors.append("Maximum agents per simulation is too high")
        
        if self.max_concurrent_simulations > 1000:
            errors.append("Maximum concurrent simulations is too high")
        
        # Check scaling configuration
        if self.scaling.min_replicas < 2:
            errors.append("Minimum replicas should be at least 2 for high availability")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses
        
        def dict_factory(field_pairs):
            """Custom dict factory to handle nested dataclasses."""
            result = {}
            for key, value in field_pairs:
                if dataclasses.is_dataclass(value):
                    result[key] = dataclasses.asdict(value, dict_factory=dict_factory)
                else:
                    result[key] = value
            return result
        
        return dataclasses.asdict(self, dict_factory=dict_factory)


# Environment-specific configurations

def get_development_config() -> ProductionConfig:
    """Get development environment configuration."""
    config = ProductionConfig()
    
    config.environment = "development"
    config.debug = True
    config.workers = 1
    
    # Relaxed security for development
    config.security.bcrypt_rounds = 4
    config.security.cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Local database
    config.database.host = "localhost"
    config.database.password = "dev_password"
    
    # Disable some production features
    config.features["distributed_computing"] = False
    config.features["auto_scaling"] = False
    
    return config


def get_staging_config() -> ProductionConfig:
    """Get staging environment configuration."""
    config = ProductionConfig()
    
    config.environment = "staging"
    config.debug = False
    config.workers = 2
    
    # Staging-specific settings
    config.max_agents_per_simulation = 5000
    config.max_concurrent_simulations = 50
    
    # Enable all features for testing
    for feature in config.features:
        config.features[feature] = True
    
    return config


def get_production_config() -> ProductionConfig:
    """Get production environment configuration."""
    return ProductionConfig.from_environment()


# Configuration factory
def get_config(environment: Optional[str] = None) -> ProductionConfig:
    """Get configuration for specified environment.
    
    Args:
        environment: Environment name (development, staging, production)
        
    Returns:
        Configuration object
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "development":
        return get_development_config()
    elif environment == "staging":
        return get_staging_config()
    elif environment == "production":
        return get_production_config()
    else:
        raise ValueError(f"Unknown environment: {environment}")


# Global configuration instance
config = get_config()