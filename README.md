# agent-swarm-eval-arena

🏟️ **Real-Time Arena for Multi-Agent Reinforcement Learning with 1000+ Concurrent Agents**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ray](https://img.shields.io/badge/Ray-2.0+-red.svg)](https://ray.io/)
[![Arena](https://img.shields.io/badge/Agents-1000+-green.svg)](https://github.com/yourusername/agent-swarm-eval-arena)

## Overview

The agent-swarm-eval-arena provides a massively scalable evaluation platform for Multi-Agent Reinforcement Learning (MARL) research. Supporting 1000+ concurrent agents with real-time telemetry, fairness probes, and comprehensive benchmarking—addressing the explosion of open-source agent frameworks in 2025.

## Key Features

- **Massive Scale**: Test with 1000+ agents simultaneously
- **Real-Time Telemetry**: Stream metrics via WebSocket with <10ms latency
- **Fairness Monitoring**: Built-in probes for emergent cooperation/competition
- **Environment Zoo**: 50+ pre-built environments from simple to complex
- **Distributed Execution**: Ray-based scaling across multiple GPUs/nodes
- **Reproducible Evaluation**: Deterministic seeding and replay capabilities
- **Live Visualization**: Interactive 3D arena viewer with agent trajectories

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/agent-swarm-eval-arena.git
cd agent-swarm-eval-arena

# Create conda environment
conda create -n swarm-arena python=3.9
conda activate swarm-arena

# Install dependencies
pip install -e .

# Install with visualization support
pip install -e ".[viz]"

# Install distributed execution support
pip install -e ".[distributed]"
```

## Quick Start

### 1. Create a Simple Swarm Evaluation

```python
from swarm_arena import Arena, SwarmConfig, Agent

# Define agent types
class CooperativeAgent(Agent):
    def act(self, observation):
        # Cooperative strategy
        nearby_agents = observation["nearby_agents"]
        if len(nearby_agents) > 0:
            # Move towards center of nearby agents
            return self.move_towards(nearby_agents.mean())
        return self.random_action()

class CompetitiveAgent(Agent):
    def act(self, observation):
        # Competitive strategy
        resources = observation["resources"]
        if len(resources) > 0:
            # Move to nearest resource
            return self.move_towards(resources[0])
        return self.random_action()

# Configure arena
config = SwarmConfig(
    num_agents=100,
    arena_size=(1000, 1000),
    episode_length=1000,
    resource_spawn_rate=0.1
)

# Create arena with mixed agent types
arena = Arena(config)
arena.add_agents(CooperativeAgent, count=50)
arena.add_agents(CompetitiveAgent, count=50)

# Run evaluation
results = arena.evaluate(
    num_episodes=10,
    metrics=["efficiency", "fairness", "emergence"],
    record_trajectories=True
)

print(f"Average reward: {results.mean_reward:.2f}")
print(f"Fairness index: {results.fairness_index:.2f}")
print(f"Emergent behaviors: {results.emergent_patterns}")
```

### 2. Large-Scale Distributed Evaluation

```python
from swarm_arena import DistributedArena
import ray

# Initialize Ray cluster
ray.init(address="ray://head-node:10001")

# Configure massive swarm
config = SwarmConfig(
    num_agents=5000,
    arena_size=(10000, 10000),
    episode_length=5000,
    physics_engine="mujoco",
    collision_detection=True
)

# Create distributed arena
arena = DistributedArena(
    config,
    num_workers=32,
    gpus_per_worker=0.25
)

# Define evaluation scenarios
scenarios = [
    {"name": "resource_scarcity", "resources": 100},
    {"name": "resource_abundance", "resources": 10000},
    {"name": "predator_prey", "predators": 500, "prey": 4500}
]

# Run parallel evaluations
results = arena.evaluate_scenarios(
    scenarios,
    metrics=["survival_rate", "resource_efficiency", "social_welfare"],
    aggregate_stats=True
)

# Analyze emergent behavior
from swarm_arena.analysis import EmergenceAnalyzer

analyzer = EmergenceAnalyzer()
patterns = analyzer.detect_patterns(
    results.trajectories,
    methods=["clustering", "flocking", "hierarchy_formation"]
)

print(f"Detected patterns: {patterns}")
```

### 3. Real-Time Monitoring Dashboard

```python
from swarm_arena import RealtimeMonitor

# Start monitoring server
monitor = RealtimeMonitor(
    port=8080,
    update_frequency=30  # Hz
)

# Connect arena to monitor
arena.attach_monitor(monitor)

# Launch web dashboard
monitor.launch_dashboard()

# Access at http://localhost:8080
# Features:
# - Live agent positions
# - Performance metrics
# - Resource distribution heatmap
# - Interaction graphs
# - Fairness metrics over time
```

### 4. Agent Tournament System

```python
from swarm_arena import Tournament, EloRating

# Create tournament with multiple agent implementations
tournament = Tournament(
    arena_config=config,
    rating_system=EloRating(initial_rating=1500)
)

# Register agent teams
tournament.register_team("DeepRL Lab", "path/to/deeprl_agents.py")
tournament.register_team("Evolutionary AI", "path/to/evo_agents.py")
tournament.register_team("Hybrid Swarm", "path/to/hybrid_agents.py")

# Run round-robin tournament
results = tournament.run_round_robin(
    matches_per_pair=10,
    environments=["foraging", "territory_control", "pursuit_evasion"]
)

# Display leaderboard
leaderboard = tournament.get_leaderboard()
print(leaderboard.to_markdown())
```

## Architecture

```
agent-swarm-eval-arena/
├── swarm_arena/
│   ├── core/
│   │   ├── arena.py              # Main arena logic
│   │   ├── agent.py              # Base agent class
│   │   ├── environment.py        # Environment interface
│   │   └── physics.py            # Physics engines
│   ├── distributed/
│   │   ├── ray_arena.py          # Ray-based distribution
│   │   ├── worker.py             # Worker processes
│   │   └── synchronization.py   # State sync
│   ├── environments/
│   │   ├── foraging.py           # Resource collection
│   │   ├── pursuit_evasion.py   # Predator-prey
│   │   ├── territory.py          # Territory control
│   │   ├── traffic.py            # Traffic simulation
│   │   └── custom_loader.py      # Load custom envs
│   ├── monitoring/
│   │   ├── telemetry.py          # Metrics collection
│   │   ├── streaming.py          # WebSocket server
│   │   ├── dashboard/            # Web dashboard
│   │   └── replay.py             # Replay system
│   ├── analysis/
│   │   ├── fairness.py           # Fairness metrics
│   │   ├── emergence.py          # Pattern detection
│   │   ├── statistics.py         # Statistical analysis
│   │   └── visualization.py      # Plotting tools
│   └── benchmarks/
│       ├── standard_agents.py    # Baseline agents
│       ├── metrics.py            # Evaluation metrics
│       └── leaderboard.py        # Competition system
├── environments/
│   ├── configs/                  # Environment configs
│   ├── assets/                   # 3D models, textures
│   └── scenarios/                # Predefined scenarios
├── examples/
│   ├── tutorials/                # Step-by-step guides
│   ├── research/                 # Research examples
│   └── competitions/             # Competition setups
├── tests/
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── performance/              # Performance benchmarks
└── web/
    ├── frontend/                 # React dashboard
    ├── api/                      # REST/WebSocket API
    └── static/                   # Static assets
```

## Environments

### Built-in Environment Suite

| Environment | Description | Agents | Metrics | Complexity |
|-------------|-------------|--------|---------|------------|
| Foraging | Collect distributed resources | 10-1000 | Efficiency, fairness | Low |
| Pursuit-Evasion | Predators chase prey | 20-500 | Capture rate, survival | Medium |
| Territory Control | Claim and defend areas | 50-2000 | Territory%, conflicts | Medium |
| Traffic Flow | Navigate intersections | 100-5000 | Throughput, collisions | High |
| Market Trading | Buy/sell with pricing | 50-1000 | Profit, market stability | High |
| Swarm Construction | Build structures together | 100-2000 | Build speed, coordination | Very High |

### Custom Environment Creation

```python
from swarm_arena import Environment, Entity
import numpy as np

class EpidemicEnvironment(Environment):
    """Custom environment for epidemic spreading dynamics"""
    
    def __init__(self, config):
        super().__init__(config)
        self.infection_radius = config.get("infection_radius", 10)
        self.infection_probability = config.get("infection_prob", 0.1)
        self.recovery_time = config.get("recovery_time", 100)
        
    def reset(self):
        # Initialize agent states
        self.agent_states = {
            agent_id: {
                "infected": np.random.random() < 0.05,  # 5% initially infected
                "infection_time": 0,
                "immune": False
            }
            for agent_id in self.agents
        }
        return self.get_observations()
    
    def step(self, actions):
        # Execute agent movements
        self.execute_actions(actions)
        
        # Update infection dynamics
        self._spread_infection()
        self._update_recovery()
        
        # Calculate rewards (minimize spread)
        rewards = self._calculate_rewards()
        
        return self.get_observations(), rewards, self.done, {}
    
    def _spread_infection(self):
        positions = self.get_agent_positions()
        
        for agent_id, pos in positions.items():
            if self.agent_states[agent_id]["infected"]:
                # Find nearby agents
                nearby = self.get_agents_in_radius(pos, self.infection_radius)
                
                for other_id in nearby:
                    if not self.agent_states[other_id]["infected"] and \
                       not self.agent_states[other_id]["immune"]:
                        if np.random.random() < self.infection_probability:
                            self.agent_states[other_id]["infected"] = True
                            self.agent_states[other_id]["infection_time"] = 0

# Register custom environment
from swarm_arena import register_environment

register_environment("epidemic", EpidemicEnvironment)
```

## Advanced Features

### Hierarchical Multi-Agent Systems

```python
from swarm_arena import HierarchicalArena, Commander, Soldier

class StrategicCommander(Commander):
    """High-level strategic decision maker"""
    
    def __init__(self, team_size=10):
        super().__init__(team_size)
        self.strategy_net = self.build_strategy_network()
    
    def plan(self, global_state):
        # Generate high-level strategy
        strategy = self.strategy_net(global_state)
        
        # Assign objectives to soldiers
        objectives = self.decompose_strategy(strategy)
        return objectives

class TacticalSoldier(Soldier):
    """Low-level tactical executor"""
    
    def act(self, observation, commander_objective):
        # Combine local observation with commander's objective
        action = self.policy_net(observation, commander_objective)
        return action

# Create hierarchical system
hierarchical_arena = HierarchicalArena(
    num_teams=4,
    team_size=50,
    command_structure="centralized"  # or "decentralized"
)

# Add teams with different command structures
hierarchical_arena.add_team(
    commander_class=StrategicCommander,
    soldier_class=TacticalSoldier,
    team_name="Strategic Forces"
)
```

### Fairness and Emergent Behavior Analysis

```python
from swarm_arena.analysis import FairnessProbe, EmergenceDetector

# Configure fairness metrics
fairness_probe = FairnessProbe(
    metrics=[
        "gini_coefficient",      # Resource distribution inequality
        "envy_freeness",         # No agent prefers another's allocation
        "proportional_fairness", # Resources proportional to contribution
        "max_min_fairness"       # Maximize minimum agent welfare
    ]
)

# Attach to arena
arena.attach_probe(fairness_probe)

# Run experiment
results = arena.run(episodes=100)

# Analyze fairness over time
fairness_report = fairness_probe.generate_report()
print(f"Gini coefficient: {fairness_report['gini_coefficient']:.3f}")
print(f"Envy-free episodes: {fairness_report['envy_free_ratio']:.2%}")

# Detect emergent behaviors
emergence_detector = EmergenceDetector()
patterns = emergence_detector.analyze(results.trajectories)

for pattern in patterns:
    print(f"\nPattern: {pattern.name}")
    print(f"Confidence: {pattern.confidence:.2%}")
    print(f"Agents involved: {len(pattern.agents)}")
    print(f"Duration: {pattern.duration} steps")
    
    # Visualize pattern
    pattern.visualize(save_path=f"patterns/{pattern.name}.mp4")
```

### Communication Protocols

```python
from swarm_arena import CommunicationChannel, Message

# Define communication protocol
class NegotiationProtocol:
    def __init__(self, vocab_size=10, message_length=5):
        self.vocab_size = vocab_size
        self.message_length = message_length
    
    def encode_proposal(self, resource_split):
        # Encode resource allocation proposal
        return Message(
            type="proposal",
            content=self.discretize(resource_split),
            sender=self.agent_id
        )
    
    def decode_response(self, message):
        if message.type == "accept":
            return True
        elif message.type == "counter":
            return self.parse_counter_offer(message.content)
        return False

# Create arena with communication
comm_arena = Arena(
    config=config,
    communication=CommunicationChannel(
        protocol=NegotiationProtocol(),
        range=50,  # Communication radius
        bandwidth=10,  # Messages per step
        noise=0.1  # Message corruption probability
    )
)

# Agents can now negotiate
class NegotiatingAgent(Agent):
    def act(self, observation):
        # Check for messages
        messages = observation.get("messages", [])
        
        for msg in messages:
            if msg.type == "proposal":
                # Evaluate and respond
                response = self.evaluate_proposal(msg)
                self.send_message(response)
        
        # Regular action
        return self.base_policy(observation)
```

### Performance Profiling

```python
from swarm_arena.profiling import PerformanceProfiler

# Create profiler
profiler = PerformanceProfiler(
    metrics=["fps", "cpu_usage", "memory_usage", "gpu_usage"],
    sampling_rate=10  # Hz
)

# Profile different configurations
configs_to_test = [
    {"num_agents": 100, "physics": "simple"},
    {"num_agents": 1000, "physics": "simple"},
    {"num_agents": 100, "physics": "mujoco"},
    {"num_agents": 1000, "physics": "mujoco"}
]

results = {}
for config in configs_to_test:
    arena = Arena(SwarmConfig(**config))
    
    with profiler.profile() as p:
        arena.run(episodes=10)
    
    results[str(config)] = p.get_summary()

# Generate performance report
profiler.generate_report(
    results,
    save_path="performance_report.html",
    include_plots=True
)
```

## Benchmarking Suite

### Standard Benchmarks

```python
from swarm_arena.benchmarks import StandardBenchmark

# Run comprehensive benchmark
benchmark = StandardBenchmark()
results = benchmark.run_all(
    agent_classes=[
        YourAgent,
        CooperativeAgent,
        CompetitiveAgent,
        RandomAgent
    ],
    environments=[
        "foraging_sparse",
        "foraging_dense", 
        "pursuit_evasion",
        "territory_control"
    ],
    metrics=["reward", "efficiency", "coordination", "adaptation"],
    num_seeds=10
)

# Generate LaTeX table for paper
latex_table = benchmark.to_latex(
    results,
    caption="Performance comparison across environments",
    label="tab:benchmark_results"
)

print(latex_table)
```

### Scaling Analysis

```python
from swarm_arena.benchmarks import ScalingBenchmark

# Test scaling properties
scaling_bench = ScalingBenchmark()

# Weak scaling: increase agents and arena size proportionally
weak_scaling = scaling_bench.weak_scaling_test(
    base_agents=100,
    base_arena=1000,
    scale_factors=[1, 2, 4, 8, 16, 32],
    metric="throughput"
)

# Strong scaling: fixed problem size, more compute
strong_scaling = scaling_bench.strong_scaling_test(
    num_agents=1000,
    num_workers=[1, 2, 4, 8, 16, 32],
    metric="episode_time"
)

# Plot scaling efficiency
scaling_bench.plot_scaling_curves(
    weak_scaling,
    strong_scaling,
    save_path="scaling_analysis.pdf"
)
```

## Integration with Popular Frameworks

### RLlib Integration

```python
from ray import tune
from ray.rllib.agents import ppo
from swarm_arena.integrations import RLlibSwarmEnv

# Wrap arena for RLlib
def env_creator(env_config):
    return RLlibSwarmEnv(
        arena_config=env_config["arena_config"],
        agent_id=env_config.worker_index
    )

# Register environment
tune.register_env("swarm_arena", env_creator)

# Configure PPO training
config = ppo.DEFAULT_CONFIG.copy()
config.update({
    "env": "swarm_arena",
    "env_config": {
        "arena_config": {
            "num_agents": 100,
            "episode_length": 1000
        }
    },
    "num_workers": 16,
    "num_gpus": 1,
    "framework": "torch"
})

# Train agents
tune.run(
    "PPO",
    config=config,
    stop={"training_iteration": 1000},
    checkpoint_freq=10
)
```

### Stable Baselines3 Integration

```python
from stable_baselines3 import PPO, A2C
from swarm_arena.integrations import SB3SwarmWrapper

# Create wrapped environment
env = SB3SwarmWrapper(
    arena_config=config,
    flatten_observation=True,
    normalize_reward=True
)

# Train with SB3
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4
)

model.learn(total_timesteps=1_000_000)
```

## Visualization and Analysis

### 3D Arena Viewer

```python
from swarm_arena.visualization import Arena3DViewer

# Create interactive 3D viewer
viewer = Arena3DViewer(
    resolution=(1920, 1080),
    camera_mode="free",  # or "follow", "overview"
    render_trails=True,
    trail_length=50
)

# Load and replay episode
episode_data = arena.load_episode("episodes/best_episode.pkl")
viewer.replay(
    episode_data,
    playback_speed=1.0,
    save_video="episode_replay.mp4"
)

# Live visualization during training
arena.attach_viewer(viewer)
arena.run(episodes=10, render=True)
```

### Statistical Analysis

```python
from swarm_arena.analysis import StatisticalAnalyzer
import pandas as pd

# Load experiment results
analyzer = StatisticalAnalyzer()
data = analyzer.load_experiments("experiments/")

# Perform statistical tests
results = analyzer.compare_agents(
    data,
    test="friedman",  # Non-parametric test for multiple comparisons
    post_hoc="nemenyi",
    significance_level=0.05
)

# Generate publication-ready plots
analyzer.create_figures(
    data,
    figures=[
        "learning_curves",
        "final_performance_boxplot", 
        "critical_difference_diagram",
        "emergence_heatmap"
    ],
    style="publication",
    save_dir="figures/"
)
```

## Best Practices

### Experiment Reproducibility

```python
from swarm_arena import set_global_seed, ExperimentLogger

# Ensure reproducibility
set_global_seed(42)

# Log all experiment details
logger = ExperimentLogger(
    experiment_name="swarm_cooperation_study",
    log_dir="experiments/2025_01_15/",
    auto_git_commit=True  # Commit code state
)

# Log hyperparameters
logger.log_hyperparameters({
    "num_agents": 500,
    "learning_rate": 1e-4,
    "architecture": "transformer",
    "communication": True
})

# Run experiment with automatic logging
with logger.run():
    results = arena.evaluate(episodes=100)
    logger.log_metrics(results.to_dict())
```

### Performance Optimization

```python
# Use efficient data structures
from swarm_arena.optimizations import CompactStateRepresentation

# Enable optimizations
arena = Arena(
    config,
    optimizations={
        "state_representation": CompactStateRepresentation(),
        "parallel_environments": 32,
        "vectorized_actions": True,
        "gpu_acceleration": True,
        "compile_mode": "torch.compile"  # PyTorch 2.0
    }
)

# Profile bottlenecks
from swarm_arena.profiling import profile_arena

profile_report = profile_arena(
    arena,
    num_steps=10000,
    profile_gpu=True
)

print(f"Bottleneck: {profile_report.bottleneck}")
print(f"Suggested optimization: {profile_report.suggestion}")
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Research Collaborations

We're particularly interested in:
- Novel emergent behavior metrics
- Efficient multi-agent communication protocols
- Fairness-aware MARL algorithms
- Large-scale (10,000+ agents) optimizations

## Citation

```bibtex
@software{agent_swarm_eval_arena,
  title = {Agent Swarm Evaluation Arena: A Scalable Platform for Multi-Agent Research},
  author = {Daniel Schmidt},
  year = {2025},
  url = {https://github.com/danieleschmidt/agent-swarm-eval-arena}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Ray team for distributed computing framework
- OpenAI for multi-agent research insights
- The broader MARL community for benchmark contributions
