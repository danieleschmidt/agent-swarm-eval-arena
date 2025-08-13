# API Reference

## Table of Contents

1. [Core Components](#core-components)
2. [Agent Types](#agent-types)
3. [Research Tools](#research-tools)
4. [Security & Authentication](#security--authentication)
5. [Reliability & Monitoring](#reliability--monitoring)
6. [Optimization & Scaling](#optimization--scaling)
7. [Configuration](#configuration)

## Core Components

### Arena

The main simulation environment that orchestrates agent interactions.

```python
from swarm_arena import Arena, SwarmConfig

# Basic usage
config = SwarmConfig(num_agents=100, arena_size=(1000, 1000))
arena = Arena(config)
arena.add_agents(CooperativeAgent, count=50)
arena.add_agents(CompetitiveAgent, count=50)
results = arena.run(episodes=10)
```

**Methods:**
- `add_agents(agent_class, count)`: Add agents to the arena
- `run(episodes, verbose=False)`: Execute simulation
- `reset()`: Reset arena to initial state
- `get_metrics()`: Get real-time simulation metrics

### SwarmConfig

Configuration object for arena and simulation parameters.

```python
config = SwarmConfig(
    num_agents=100,
    arena_size=(1000, 1000),
    episode_length=1000,
    max_agent_speed=5.0,
    observation_radius=50.0,
    collision_detection=True,
    seed=42
)
```

**Parameters:**
- `num_agents`: Number of agents in simulation
- `arena_size`: Tuple of (width, height) for arena dimensions
- `episode_length`: Maximum steps per episode
- `max_agent_speed`: Maximum movement speed for agents
- `observation_radius`: Agent perception range
- `collision_detection`: Enable/disable collision physics
- `seed`: Random seed for reproducibility

## Agent Types

### CooperativeAgent

Agents that collaborate and share resources.

```python
from swarm_arena import CooperativeAgent

agent = CooperativeAgent(agent_id=0, position=[100, 100])
```

**Behaviors:**
- Resource sharing with nearby agents
- Collective task coordination
- Emergence of leadership roles

### CompetitiveAgent

Agents that compete for resources and territory.

```python
from swarm_arena import CompetitiveAgent

agent = CompetitiveAgent(agent_id=1, position=[200, 200])
```

**Behaviors:**
- Territorial control
- Resource hoarding
- Strategic positioning

### CustomAgent

Base class for implementing custom agent behaviors.

```python
from swarm_arena import BaseAgent

class MyAgent(BaseAgent):
    def act(self, observation):
        # Custom decision logic
        return self.choose_action(observation)
    
    def update(self, reward, done):
        # Custom learning logic
        pass
```

## Research Tools

### EmergenceDetector

Analyzes agent trajectories for emergent patterns.

```python
from swarm_arena.research import EmergenceDetector

detector = EmergenceDetector()
trajectories = arena.get_agent_trajectories()
patterns = detector.analyze(trajectories)

for pattern in patterns:
    print(f"Pattern: {pattern.pattern_type}")
    print(f"Strength: {pattern.strength}")
    print(f"Agents involved: {pattern.agent_ids}")
```

**Pattern Types:**
- `FLOCKING`: Coordinated movement patterns
- `CLUSTERING`: Spatial grouping behaviors
- `LEADERSHIP`: Follower-leader dynamics
- `SPECIALIZATION`: Role differentiation

### FairnessAnalyzer

Measures resource distribution fairness across agents.

```python
from swarm_arena.research import FairnessAnalyzer

analyzer = FairnessAnalyzer()
rewards = arena.get_agent_rewards()

# Calculate fairness metrics
gini = analyzer.gini_coefficient(rewards)
jain = analyzer.jain_index(rewards)
envy_free = analyzer.is_envy_free(rewards, agent_preferences)

print(f"Gini coefficient: {gini:.3f}")
print(f"Jain's index: {jain:.3f}")
print(f"Envy-free: {envy_free}")
```

### MessageChannel

Enables agent communication with realistic constraints.

```python
from swarm_arena.research.communication import MessageChannel, Message

channel = MessageChannel(
    max_range=100.0,
    bandwidth_limit=10,
    noise_probability=0.1
)

# Send message
message = Message(
    sender_id=0,
    receiver_id=1,
    message_type=MessageType.PROPOSAL,
    content={"resource_split": [0.6, 0.4]}
)

success = channel.send_message(message, agent_positions, current_step)
```

## Security & Authentication

### AuthenticationManager

Manages user authentication and authorization.

```python
from swarm_arena.security import AuthenticationManager, UserRole

auth = AuthenticationManager()

# Create user
user = auth.create_user(
    username="researcher",
    email="user@example.com",
    password="secure_password",
    role=UserRole.RESEARCHER
)

# Authenticate
token = auth.authenticate("researcher", "secure_password")

# Check permissions
can_create = auth.check_permission(token, Permission.ARENA_CREATE)
```

### InputSanitizer

Validates and sanitizes user input to prevent attacks.

```python
from swarm_arena.security import InputSanitizer, ConfigValidator

sanitizer = InputSanitizer()
validator = ConfigValidator()

# Sanitize string input
clean_input = sanitizer.sanitize_string(user_input)

# Validate configuration
try:
    valid_config = validator.validate_config(config_dict)
except ValidationError as e:
    print(f"Invalid config: {e}")
```

## Reliability & Monitoring

### Circuit Breaker

Implements circuit breaker pattern for fault tolerance.

```python
from swarm_arena.reliability import circuit_breaker

@circuit_breaker("external_service", failure_threshold=3, recovery_timeout=30.0)
def unreliable_operation():
    # Operation that might fail
    return external_api_call()

try:
    result = unreliable_operation()
except CircuitBreakerError:
    # Handle circuit breaker open state
    result = fallback_operation()
```

### Retry Mechanism

Automatic retry with configurable strategies.

```python
from swarm_arena.reliability import retry, RetryStrategy

@retry(max_attempts=3, base_delay=1.0, strategy=RetryStrategy.EXPONENTIAL_BACKOFF)
def flaky_operation():
    # Operation that might temporarily fail
    return api_call()
```

### Metrics Collection

Advanced telemetry and monitoring.

```python
from swarm_arena.monitoring import metrics_collector, time_operation

# Record metrics
metrics_collector.record_counter("requests.total", 1)
metrics_collector.record_gauge("cpu.usage", 75.5)
metrics_collector.record_histogram("response.time", 0.25)

# Time operations
@time_operation("simulation.step")
def simulation_step():
    # Simulation logic
    pass
```

## Optimization & Scaling

### Performance Optimization

Vectorized operations and caching for better performance.

```python
from swarm_arena.optimization import VectorizedOperations, AdaptiveCache

# Vectorized distance calculations
positions = np.array([[x1, y1], [x2, y2], ...])
distances = VectorizedOperations.distance_matrix(positions)

# Adaptive caching
cache = AdaptiveCache(max_size=1000, ttl=300)
result = cache.get_or_compute("expensive_calc", expensive_function, args)
```

### Auto-scaling

Dynamic resource management based on load.

```python
from swarm_arena.optimization import AutoScaler, ScalingPolicy

policy = ScalingPolicy(
    min_instances=2,
    max_instances=10,
    scale_up_threshold=80.0,
    scale_down_threshold=20.0
)

scaler = AutoScaler(policy)
scaling_decision = scaler.should_scale(current_metrics)
```

## Configuration

### Production Configuration

Environment-specific configuration management.

```python
from deployment.production_config import get_config

# Get configuration for current environment
config = get_config()  # Uses ENVIRONMENT env var

# Or specify environment
dev_config = get_config("development")
prod_config = get_config("production")

# Validate configuration
errors = config.validate()
if errors:
    print("Configuration errors:", errors)
```

### Quality Gates

Automated quality validation.

```python
from quality_gates import QualityGateRunner

runner = QualityGateRunner()
all_passed = runner.run_all_gates()

if all_passed:
    print("✅ All quality gates passed - ready for production")
else:
    print("❌ Quality gates failed - review required")
```

## CLI Interface

Command-line interface for easy interaction.

```bash
# Run simulation
swarm-arena simulate --agents 100 --episodes 5 --verbose

# Run benchmarks
swarm-arena benchmark --seeds 10 --output results.json

# Scaling tests
swarm-arena scaling --test-type weak --scale-factors 1 2 4 8

# Create example config
swarm-arena config --output example.json
```

## Error Handling

The framework provides comprehensive error handling:

```python
from swarm_arena.exceptions import (
    SwarmArenaError,
    ConfigurationError,
    SimulationError,
    AuthenticationError
)

try:
    arena = Arena(config)
    results = arena.run()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except SimulationError as e:
    print(f"Simulation error: {e}")
except SwarmArenaError as e:
    print(f"General error: {e}")
```

## Events and Callbacks

Register callbacks for simulation events:

```python
from swarm_arena.events import EventManager

event_manager = EventManager()

@event_manager.on("agent_collision")
def handle_collision(event_data):
    print(f"Collision between agents {event_data['agent1']} and {event_data['agent2']}")

@event_manager.on("episode_complete")
def handle_episode_end(event_data):
    print(f"Episode {event_data['episode']} completed with reward {event_data['reward']}")
```

## Advanced Features

### Distributed Computing

Scale across multiple nodes using Ray:

```python
import ray
from swarm_arena.distributed import DistributedArena

ray.init()
arena = DistributedArena(config, num_workers=4)
results = arena.run_distributed(episodes=100)
```

### Real-time Visualization

Monitor simulations in real-time:

```python
from swarm_arena.visualization import RealtimeVisualizer

visualizer = RealtimeVisualizer(arena)
visualizer.start()  # Opens web interface
```

### Experiment Tracking

Track and reproduce experiments:

```python
from swarm_arena.research import ExperimentLogger

logger = ExperimentLogger("experiment_001")
logger.log_config(config)
logger.log_results(results)
logger.save_checkpoint(arena.get_state())
```