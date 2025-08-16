# Quick Start Guide - SwarmArena

Welcome to SwarmArena, the next-generation multi-agent reinforcement learning platform with breakthrough quantum and neuromorphic capabilities.

## 🚀 Installation

### Prerequisites
- Python 3.9+
- 4GB+ RAM recommended
- Optional: CUDA-capable GPU for acceleration

### Basic Installation
```bash
# Clone repository
git clone https://github.com/danieleschmidt/agent-swarm-eval-arena.git
cd agent-swarm-eval-arena

# Install dependencies
pip install -e .

# For full features (neural networks)
pip install torch torchvision

# For visualization support
pip install -e ".[viz]"

# For distributed computing
pip install -e ".[distributed]"
```

### Docker Installation
```bash
# Build container
docker build -t swarm-arena .

# Run with basic setup
docker run -p 8080:8080 swarm-arena

# Run with GPU support
docker run --gpus all -p 8080:8080 swarm-arena
```

## 🎯 Basic Usage

### Simple Swarm Simulation
```python
from swarm_arena import Arena, SwarmConfig, CooperativeAgent

# Configure arena
config = SwarmConfig(
    num_agents=50,
    arena_size=(1000, 1000),
    episode_length=1000
)

# Create arena and add agents
arena = Arena(config)
arena.add_agents(CooperativeAgent, count=50)

# Run simulation
results = arena.evaluate(
    num_episodes=10,
    metrics=["efficiency", "fairness", "emergence"]
)

print(f"Mean reward: {results.mean_reward:.2f}")
print(f"Fairness index: {results.fairness_index:.2f}")
```

### Advanced Research Features (Generation 1)
```python
from swarm_arena.research.breakthrough_algorithms import BreakthroughAlgorithms

# Initialize breakthrough algorithms
breakthrough = BreakthroughAlgorithms()

# Discover causal relationships
causal_graph = breakthrough.discover_causal_structure(
    agent_trajectories, time_window=50
)

# Detect emergent behaviors
patterns = breakthrough.detect_emergent_behaviors(
    agent_positions, agent_actions
)

# Quantum fairness analysis
fairness_results = breakthrough.quantum_fairness_analysis(
    agent_rewards, agent_contributions
)
```

### Security & Authentication (Generation 2)
```python
from swarm_arena.security.advanced_authentication import AdvancedAuthenticator

# Initialize authentication
auth = AdvancedAuthenticator()

# Generate agent credentials
credentials = auth.generate_agent_credentials(agent_id=1)

# Authenticate agent
token = auth.authenticate_agent(
    agent_id=1,
    credentials=credentials,
    requested_permissions=['read_basic', 'write_basic']
)

# Validate token
valid_token = auth.validate_token(token.token_id)
```

### Quantum Optimization (Generation 3)
```python
from swarm_arena.optimization.quantum_computing_interface import QuantumOptimizer

# Initialize quantum optimizer
quantum_opt = QuantumOptimizer(backend="simulator")

# Solve agent coordination problem
result = quantum_opt.solve_agent_coordination(
    agent_positions=positions,
    target_formation=target
)

print(f"Quantum advantage: {result.quantum_advantage:.2f}x")
print(f"Solution energy: {result.energy:.3f}")
```

### Neuromorphic Processing (Generation 3)
```python
from swarm_arena.optimization.neuromorphic_processing import NeuromorphicSwarmProcessor

# Initialize neuromorphic processor
processor = NeuromorphicSwarmProcessor(
    max_agents=1000,
    processing_cores=4
)

# Process swarm behavior
result = processor.process_swarm_behavior(
    agent_positions, agent_velocities
)

print(f"Energy efficiency: {result['neuromorphic_advantages']['energy_efficiency']:.2f}x")
print(f"Behavior classification: {result['behavioral_analysis']['behavior_classification']}")
```

## 🔧 Configuration

### Arena Configuration
```python
config = SwarmConfig(
    num_agents=100,                    # Number of agents
    arena_size=(1000, 1000),          # Arena dimensions
    episode_length=1000,              # Steps per episode
    resource_spawn_rate=0.1,          # Resource generation rate
    physics_engine="simple",          # Physics engine type
    communication_range=50.0,         # Agent communication range
    reward_config={                   # Reward structure
        "resource_collection": 1.0,
        "survival_bonus": 0.01,
        "cooperation_bonus": 0.5
    }
)
```

### Security Configuration
```python
# Advanced authentication setup
authenticator = AdvancedAuthenticator()

# Configure security policies
authenticator.max_failed_attempts = 3
authenticator.lockout_duration = 300
authenticator.token_lifetime = 3600
```

### Performance Configuration
```python
# Neuromorphic processing setup
processor = NeuromorphicSwarmProcessor(
    max_agents=1000,
    processing_cores=8
)

# Quantum optimization setup
quantum_optimizer = QuantumOptimizer(backend="simulator")
```

## 📊 Monitoring & Visualization

### Real-time Monitoring
```python
from swarm_arena.monitoring.streaming import StreamingServer

# Start monitoring server
server = StreamingServer(port=8080)
arena.attach_monitor(server)

# Access dashboard at http://localhost:8080
server.start()
```

### Performance Metrics
```python
from swarm_arena.monitoring.telemetry import TelemetryCollector

# Enable telemetry
telemetry = TelemetryCollector()
arena.telemetry_collector = telemetry

# Get performance report
performance_report = telemetry.get_performance_summary()
```

## 🧪 Testing & Validation

### Run Quality Gates
```bash
# Run comprehensive validation
python comprehensive_quality_validation.py

# Run specific tests
python -m pytest tests/test_core.py -v
python -m pytest tests/test_security.py -v
python -m pytest tests/test_performance.py -v
```

### Benchmark Performance
```bash
# Run performance benchmarks
python examples/scaling_optimization_demo.py

# Run research demonstrations
python examples/breakthrough_research_demo.py

# Run robustness tests
python examples/enhanced_robustness_demo.py
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Install missing dependencies
pip install torch scikit-learn cryptography

# Set Python path
export PYTHONPATH=/path/to/swarm-arena:$PYTHONPATH
```

**Performance Issues**
```python
# Enable optimizations
config = SwarmConfig(
    num_agents=50,  # Reduce for testing
    episode_length=100,  # Shorter episodes
    physics_engine="simple"  # Faster physics
)

# Disable expensive features for testing
results = arena.evaluate(record_trajectories=False)
```

**Memory Issues**
```python
# Enable memory optimization
from swarm_arena.optimization.performance_engine import PerformanceOptimizer

optimizer = PerformanceOptimizer()
arena.performance_optimizer = optimizer
```

### Getting Help

- **Documentation**: Check `docs/` directory
- **Examples**: See `examples/` for working code
- **Issues**: Report bugs on GitHub
- **Community**: Join discussions in repository

## 🚀 Next Steps

1. **Explore Examples**: Run the demo scripts in `examples/`
2. **Read Documentation**: Check `docs/API_REFERENCE.md`
3. **Join Community**: Contribute to the project
4. **Deploy**: Use Docker for production deployment

## 📖 Additional Resources

- [API Reference](API_REFERENCE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Performance Benchmarks](PERFORMANCE_BENCHMARKS.md)
- [Research Findings](RESEARCH_FINDINGS.md)

---

*Happy Swarming! 🐛✨*