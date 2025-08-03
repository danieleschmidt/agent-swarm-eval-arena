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
│   └── competitions
