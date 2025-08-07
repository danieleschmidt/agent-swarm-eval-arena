# Sentiment-Aware Multi-Agent System Deployment Guide

## 🚀 Production Deployment Guide

This guide provides comprehensive instructions for deploying the Sentiment-Aware Multi-Agent Reinforcement Learning (SA-MARL) system in production environments.

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.8+
- 4GB RAM
- 2 CPU cores
- 10GB disk space

**Recommended for Production:**
- Python 3.10+
- 16GB+ RAM
- 8+ CPU cores
- 100GB+ SSD storage
- GPU support (optional, for advanced NLP models)

### Dependencies

**Core Dependencies:**
```bash
# Required system packages
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv

# Python dependencies (install via pip or system packages)
pip install numpy>=1.21.0
pip install scipy>=1.7.0
```

**Optional Dependencies for Enhanced Features:**
```bash
# Distributed computing
pip install ray[default]>=2.0.0

# Advanced NLP (for enhanced text sentiment)
pip install transformers>=4.20.0
pip install torch>=1.12.0

# Visualization and monitoring
pip install matplotlib>=3.5.0
pip install plotly>=5.10.0
pip install dash>=2.6.0
```

## 🔧 Installation

### Quick Start Installation

```bash
# Clone the repository
git clone https://github.com/danieleschmidt/sentiment-analyzer-pro.git
cd sentiment-analyzer-pro

# Create virtual environment
python3 -m venv sentiment_arena_env
source sentiment_arena_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Validate installation
python3 validate_system.py
```

### Docker Installation

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Validate installation
RUN python3 validate_system.py

EXPOSE 8000
CMD ["python3", "examples/sentiment_aware_simulation.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-arena
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-arena
  template:
    metadata:
      labels:
        app: sentiment-arena
    spec:
      containers:
      - name: sentiment-arena
        image: sentiment-arena:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        env:
        - name: ARENA_CONFIG
          value: "production"
        - name: RAY_CLUSTER_ADDRESS
          value: "ray-head-service:10001"
```

## ⚙️  Configuration

### Environment Variables

```bash
# Core Configuration
export ARENA_CONFIG="production"
export LOG_LEVEL="INFO"
export SENTIMENT_CACHE_SIZE="5000"
export MAX_AGENTS="1000"

# Distributed Computing
export RAY_CLUSTER_ADDRESS="ray://head-node:10001"
export NUM_SENTIMENT_WORKERS="8"
export NUM_CONTAGION_WORKERS="4"

# Performance Tuning
export BATCH_SIZE="100"
export CACHE_TTL_SECONDS="60"
export SPATIAL_GRID_SIZE="100"

# Internationalization
export PRIMARY_LANGUAGE="en"
export CULTURAL_CONTEXT="western_individualistic"
export ENABLE_CULTURAL_ADAPTATION="true"

# Monitoring
export ENABLE_TELEMETRY="true"
export TELEMETRY_EXPORT_INTERVAL="5.0"
export METRICS_PORT="9090"
```

### Configuration Files

**config/production.json:**
```json
{
  "arena": {
    "num_agents": 1000,
    "arena_size": [2000, 2000],
    "episode_length": 5000,
    "resource_spawn_rate": 0.05,
    "max_agent_speed": 15.0
  },
  "sentiment": {
    "enable_sentiment_processing": true,
    "enable_emotional_contagion": true,
    "contagion_influence_radius": 150.0,
    "sentiment_update_interval": 1.0
  },
  "performance": {
    "enable_caching": true,
    "cache_size": 5000,
    "enable_batch_processing": true,
    "batch_size": 100,
    "enable_spatial_optimization": true
  },
  "distributed": {
    "enable_distributed": true,
    "num_sentiment_workers": 8,
    "num_contagion_workers": 4,
    "worker_cpu_allocation": 2.0,
    "worker_memory_allocation": "4GB"
  },
  "monitoring": {
    "enable_telemetry": true,
    "telemetry_export_interval": 5.0,
    "enable_performance_tracking": true,
    "log_level": "INFO"
  }
}
```

## 🏗️ Architecture Deployment

### Single-Node Deployment

For research environments and smaller deployments:

```python
from swarm_arena import Arena, SwarmConfig
from swarm_arena.core.sentiment_aware_agent import SentimentAwareAgent
from swarm_arena.utils.performance_optimizer import PerformanceOptimizer

# Configure for single-node deployment
config = SwarmConfig()
config.num_agents = 500
config.arena_size = (1500, 1500)

# Initialize performance optimizer
optimizer = PerformanceOptimizer()
optimizer.initialize_spatial_grid(config.arena_size[0], config.arena_size[1])

# Create arena with sentiment awareness
arena = Arena(config, performance_optimizer=optimizer)

# Run simulation
arena.run_episodes(num_episodes=100)
```

### Distributed Cluster Deployment

For production and large-scale research:

```python
from swarm_arena.utils.distributed_computing import DistributedSentimentManager, DistributedConfig

# Configure distributed computing
distributed_config = DistributedConfig(
    enable_distributed=True,
    ray_address="ray://ray-head:10001",
    num_sentiment_workers=16,
    num_contagion_workers=8,
    worker_cpu_allocation=2.0,
    worker_memory_allocation="4GB"
)

# Initialize distributed manager
dist_manager = DistributedSentimentManager(distributed_config)
dist_manager.initialize()

# Create high-capacity arena
config = SwarmConfig()
config.num_agents = 5000  # Large-scale simulation
config.arena_size = (5000, 5000)

arena = Arena(config, distributed_manager=dist_manager)
arena.run_episodes(num_episodes=1000)
```

## 📊 Monitoring and Observability

### Prometheus Integration

**metrics.yaml:**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'sentiment-arena'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

### Grafana Dashboard

Key metrics to monitor:

1. **Performance Metrics:**
   - Processing latency (p50, p95, p99)
   - Cache hit rates
   - Memory utilization
   - CPU utilization

2. **Sentiment Metrics:**
   - Sentiment distribution across agents
   - Emotional contagion propagation rate
   - Agent cooperation rates
   - Behavioral adaptation metrics

3. **System Metrics:**
   - Active agent count
   - Simulation frame rate
   - Error rates
   - Resource utilization

### Health Checks

```python
def health_check():
    """Production health check endpoint."""
    checks = {
        'sentiment_processor': check_sentiment_processor(),
        'emotional_contagion': check_contagion_system(),
        'distributed_workers': check_worker_status(),
        'memory_usage': check_memory_usage(),
        'performance': check_performance_metrics()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return {
        'status': 'healthy' if all_healthy else 'unhealthy',
        'checks': checks,
        'timestamp': time.time()
    }, status_code
```

## 🔒 Security Considerations

### Input Validation

```python
def validate_agent_input(agent_data):
    """Validate agent configuration input."""
    required_fields = ['agent_id', 'initial_position']
    
    for field in required_fields:
        if field not in agent_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate position bounds
    pos = agent_data['initial_position']
    if not (isinstance(pos, list) and len(pos) == 2):
        raise ValueError("Invalid position format")
    
    # Validate numerical ranges
    if not (0 <= pos[0] <= 10000 and 0 <= pos[1] <= 10000):
        raise ValueError("Position out of valid range")

def sanitize_text_input(text):
    """Sanitize text input for sentiment analysis."""
    if not isinstance(text, str):
        raise TypeError("Text input must be string")
    
    # Remove potential injection patterns
    text = re.sub(r'[<>"\']', '', text)
    text = text[:1000]  # Limit length
    
    return text.strip()
```

### Resource Limits

```python
# Set resource limits for production
import resource

# Limit memory usage (in bytes)
resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))  # 8GB

# Limit CPU time (in seconds)
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour

# Limit number of processes
resource.setrlimit(resource.RLIMIT_NPROC, (1000, 1000))
```

## 🚨 Troubleshooting

### Common Issues and Solutions

**Issue: High Memory Usage**
```bash
# Monitor memory usage
python3 -c "
from swarm_arena.utils.performance_optimizer import PerformanceOptimizer
optimizer = PerformanceOptimizer()
print(optimizer.get_performance_stats())
"

# Solution: Adjust cache settings
export SENTIMENT_CACHE_SIZE="1000"  # Reduce cache size
export BATCH_SIZE="50"              # Reduce batch size
```

**Issue: Slow Processing**
```bash
# Check processing performance
python3 -c "
import time
from swarm_arena.sentiment.processor import SentimentProcessor

processor = SentimentProcessor()
start = time.time()
for i in range(100):
    processor.analyze_text_sentiment('test text')
print(f'Processing time: {(time.time() - start) * 10:.2f}ms per text')
"

# Solution: Enable distributed processing
export RAY_CLUSTER_ADDRESS="ray://localhost:10001"
export NUM_SENTIMENT_WORKERS="4"
```

**Issue: Ray Connection Failures**
```bash
# Check Ray cluster status
ray status

# Restart Ray if needed
ray stop
ray start --head --port=10001

# Test connection
python3 -c "
import ray
ray.init(address='ray://localhost:10001')
print('Ray connected successfully')
ray.shutdown()
"
```

### Performance Optimization

**CPU Optimization:**
```python
# Configure for CPU-bound workloads
config = PerformanceConfig(
    enable_multithreading=True,
    max_worker_threads=8,
    enable_batch_processing=True,
    batch_size=100
)
```

**Memory Optimization:**
```python
# Configure for memory-constrained environments
config = PerformanceConfig(
    sentiment_cache_size=1000,
    cache_ttl_seconds=15.0,
    max_memory_entries=5000,
    memory_cleanup_interval=50
)
```

**Network Optimization:**
```python
# Configure for distributed deployments
config = DistributedConfig(
    num_sentiment_workers=16,
    distributed_batch_size=200,
    worker_timeout=60.0,
    max_retries=5
)
```

## 📈 Scaling Guidelines

### Horizontal Scaling

**Add Ray Worker Nodes:**
```bash
# On worker nodes
ray start --address='ray-head:10001' --num-cpus=8 --memory=16000000000

# Verify cluster
ray status
```

**Scale Kubernetes Deployment:**
```bash
kubectl scale deployment sentiment-arena --replicas=10
```

### Vertical Scaling

**Increase Resource Allocation:**
```yaml
resources:
  requests:
    memory: "8Gi"
    cpu: "4000m"
  limits:
    memory: "32Gi"
    cpu: "16000m"
```

### Performance Benchmarks

**Expected Performance:**
- Single node: 1000 agents @ 60 FPS
- 4-node cluster: 5000 agents @ 60 FPS
- 16-node cluster: 20000+ agents @ 30+ FPS

**Scaling Targets:**
- Sentiment processing: <5ms per agent
- Emotional contagion: <10ms per update
- Memory usage: <4GB per 1000 agents
- Network bandwidth: <100 Mbps per node

## 🎯 Production Checklist

### Pre-Deployment
- [ ] All dependencies installed
- [ ] Configuration files validated
- [ ] Health checks implemented
- [ ] Security measures in place
- [ ] Monitoring configured
- [ ] Backup procedures established

### Deployment
- [ ] Staged deployment tested
- [ ] Load balancing configured  
- [ ] Auto-scaling enabled
- [ ] Logging aggregation active
- [ ] Alert thresholds set
- [ ] Rollback plan prepared

### Post-Deployment
- [ ] Performance metrics verified
- [ ] Error rates within limits
- [ ] User acceptance confirmed
- [ ] Documentation updated
- [ ] Team training completed
- [ ] Support procedures established

## 📚 Additional Resources

- **API Documentation:** [docs/API.md](docs/API.md)
- **Research Papers:** [docs/RESEARCH.md](docs/RESEARCH.md)
- **Performance Tuning:** [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- **Troubleshooting Guide:** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- **Development Setup:** [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

---

🎭 **Sentiment-Aware Multi-Agent System**  
Version 1.0.0 | Production Ready | Research Platform

For support: [GitHub Issues](https://github.com/danieleschmidt/sentiment-analyzer-pro/issues)