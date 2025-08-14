```python
#!/usr/bin/env python3
"""
Command Line Interface for Swarm Arena
Provides easy access to simulation capabilities from the command line.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from .core.arena import Arena
from .core.config import SwarmConfig
from .core.agent import (
    Agent, CooperativeAgent, CompetitiveAgent, RandomAgent,
    LearningAgent, HierarchicalAgent, SwarmAgent, AdaptiveAgent
)
from .core.environment import ForagingEnvironment
from .distributed.ray_arena import DistributedArena
from .monitoring.telemetry import TelemetryCollector
from .benchmarks.standard import StandardBenchmark
from .benchmarks.scaling import ScalingBenchmark
from .utils.logging import get_logger
from .utils.seeding import set_global_seed

logger = get_logger(__name__)

# Agent type mappings
AGENT_TYPES = {
    "default": Agent,
    "cooperative": CooperativeAgent,
    "competitive": CompetitiveAgent,
    "random": RandomAgent,
    "learning": LearningAgent,
    "hierarchical": HierarchicalAgent,
    "swarm": SwarmAgent,
    "adaptive": AdaptiveAgent,
}

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="🏟️  Swarm Arena - Multi-Agent Reinforcement Learning Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simple simulation with 100 agents
  swarm-arena run --agents 100 --episodes 10

  # Run with mixed agent types
  swarm-arena run --config-file experiments/mixed_agents.json

  # Run basic simulation
  swarm-arena simulate --agents 100 --episodes 5

  # Run with mixed agent types
  swarm-arena simulate --agents 200 --agent-type mixed --verbose

  # Benchmark different configurations
  swarm-arena benchmark --output results.json

  # Launch distributed simulation
  swarm-arena distributed --workers 4 --agents 1000

  # Run scaling tests
  swarm-arena scaling --test-type weak --scale-factors 1 2 4 8

  # Create example configuration
  swarm-arena config --output example_config.json
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run arena simulation")
    run_parser.add_argument("--agents", "-n", type=int, default=100,
                           help="Number of agents (default: 100)")
    run_parser.add_argument("--episodes", "-e", type=int, default=10,
                           help="Number of episodes (default: 10)")
    run_parser.add_argument("--arena-size", nargs=2, type=float, default=[1000, 1000],
                           help="Arena dimensions (default: 1000 1000)")
    run_parser.add_argument("--episode-length", type=int, default=1000,
                           help="Max steps per episode (default: 1000)")
    run_parser.add_argument("--agent-types", nargs="+", 
                           choices=list(AGENT_TYPES.keys()), default=["default"],
                           help="Agent types to use")
    run_parser.add_argument("--seed", type=int, help="Random seed")
    run_parser.add_argument("--config-file", type=str,
                           help="JSON configuration file")
    run_parser.add_argument("--output", "-o", type=str,
                           help="Output file for results")
    run_parser.add_argument("--verbose", "-v", action="store_true",
                           help="Verbose output")
    run_parser.add_argument("--telemetry", action="store_true",
                           help="Enable telemetry collection")

    # Simulate command (legacy compatibility)
    sim_parser = subparsers.add_parser("simulate", help="Run simulation")
    sim_parser.add_argument("--agents", type=int, default=100, help="Number of agents")
    sim_parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    sim_parser.add_argument("--steps", type=int, default=1000, help="Steps per episode")
    sim_parser.add_argument("--width", type=int, default=1000, help="Arena width")
    sim_parser.add_argument("--height", type=int, default=1000, help="Arena height")
    sim_parser.add_argument("--agent-type", choices=["cooperative", "competitive", "mixed", "random"],
                          default="mixed", help="Type of agents to use")
    sim_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    sim_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    sim_parser.add_argument("--output", type=str, help="Output file for results")
    sim_parser.add_argument("--config", type=str, help="Configuration file path")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    benchmark_parser.add_argument("--output", "-o", type=str, default="benchmark_results.json",
                                 help="Output file (default: benchmark_results.json)")
    benchmark_parser.add_argument("--environments", nargs="+", default=["foraging"],
                                 help="Environments to test")
    benchmark_parser.add_argument("--agent-types", nargs="+",
                                 choices=list(AGENT_TYPES.keys()),
                                 default=["default", "cooperative", "competitive"],
                                 help="Agent types to benchmark")
    benchmark_parser.add_argument("--seeds", type=int, default=5,
                                 help="Number of random seeds (default: 5)")

    # Distributed command
    distributed_parser = subparsers.add_parser("distributed", help="Run distributed simulation")
    distributed_parser.add_argument("--workers", "-w", type=int, default=4,
                                   help="Number of workers (default: 4)")
    distributed_parser.add_argument("--agents", "-n", type=int, default=1000,
                                   help="Number of agents (default: 1000)")
    distributed_parser.add_argument("--episodes", "-e", type=int, default=5,
                                   help="Number of episodes (default: 5)")
    distributed_parser.add_argument("--config-file", type=str,
                                   help="JSON configuration file")
    distributed_parser.add_argument("--ray-address", type=str,
                                   help="Ray cluster address")

    # Scaling command
    scale_parser = subparsers.add_parser("scaling", help="Run scaling tests")
    scale_parser.add_argument("--test-type", choices=["weak", "strong"], default="weak",
                            help="Type of scaling test")
    scale_parser.add_argument("--base-agents", type=int, default=100, help="Base number of agents")
    scale_parser.add_argument("--base-arena", type=int, default=1000, help="Base arena size")
    scale_parser.add_argument("--scale-factors", nargs="+", type=int, default=[1, 2, 4, 8],
                            help="Scale factors to test")
    scale_parser.add_argument("--agents", type=int, default=1000, help="Fixed number of agents for strong scaling")
    scale_parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8],
                            help="Number of workers for strong scaling")
    scale_parser.add_argument("--output", type=str, help="Output file for results")

    # Config command
    config_parser = subparsers.add_parser("config", help="Create example configuration")
    config_parser.add_argument("--output", type=str, default="config.json",
                             help="Output configuration file")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    info_parser.add_argument("--agents", action="store_true",
                           help="List available agent types")
    info_parser.add_argument("--environments", action="store_true",
                           help="List available environments")

    return parser

def load_config(config_file: str) -> SwarmConfig:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return SwarmConfig.from_dict(config_dict)
    except Exception as e:
        logger.error(f"Failed to load config from {config_file}: {e}")
        sys.exit(1)

def create_arena_from_config(config_path: str) -> Arena:
    """Create arena from configuration file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = SwarmConfig(**config_dict)
    return Arena(config)

def save_results(results: Dict[str, Any], output_file: str) -> None:
    """Save results to JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {e}")

def run_simulation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run arena simulation based on command line arguments."""
    try:
        # Load configuration
        if hasattr(args, 'config_file') and args.config_file:
            config = load_config(args.config_file)
        elif hasattr(args, 'config') and args.config:
            config = load_config(args.config)
        else:
            # Handle both new and legacy argument names
            if hasattr(args, 'arena_size'):
                arena_size = tuple(args.arena_size)
            elif hasattr(args, 'width') and hasattr(args, 'height'):
                arena_size = (args.width, args.height)
            else:
                arena_size = (1000, 1000)
            
            episode_length = getattr(args, 'episode_length', getattr(args, 'steps', 1000))
            
            config = SwarmConfig(
                num_agents=args.agents,
                arena_size=arena_size,
                episode_length=episode_length,
                seed=args.seed if hasattr(args, 'seed') else None
            )

        if args.verbose:
            print(f"📋 Configuration: {config.to_dict()}")

        # Set seed for reproducibility
        if config.seed:
            set_global_seed(config.seed)

        # Create arena
        arena = Arena(config)

        # Handle agent types
        if hasattr(args, 'agent_types'):
            # New style with multiple agent types
            agents_per_type = config.num_agents // len(args.agent_types)
            for i, agent_type in enumerate(args.agent_types):
                agent_class = AGENT_TYPES[agent_type]
                count = agents_per_type
                
                # Handle remainder for last agent type
                if i == len(args.agent_types) - 1:
                    count = config.num_agents - (agents_per_type * i)
                
                arena.add_agents(agent_class, count)
                if args.verbose:
                    print(f"Added {count} {agent_type} agents")
        elif hasattr(args, 'agent_type'):
            # Legacy style with single agent type
            if args.agent_type == "cooperative":
                arena.add_agents(CooperativeAgent, count=args.agents)
            elif args.agent_type == "competitive":
                arena.add_agents(CompetitiveAgent, count=args.agents)
            elif args.agent_type == "mixed":
                arena.add_agents(CooperativeAgent, count=args.agents // 2)
                arena.add_agents(CompetitiveAgent, count=args.agents - args.agents // 2)
            else:
                arena.add_agents(RandomAgent, count=args.agents)

        # Setup telemetry if requested
        telemetry = None
        if hasattr(args, 'telemetry') and args.telemetry:
            telemetry = TelemetryCollector()
            arena.attach_monitor(telemetry)

        # Run simulation
        print(f"🏟️  Starting simulation with {config.num_agents} agents...")
        print(f"📏 Arena size: {config.arena_size[0]}x{config.arena_size[1]}")
        print(f"⏱️  Episode length: {config.episode_length} steps")
        print(f"🔄 Running {args.episodes} episodes...")
        
        start_time = time.time()
        results = arena.run(episodes=args.episodes, verbose=args.verbose)
        end_time = time.time()
        duration = end_time - start_time

        print(f"\n📊 Simulation Results:")
        print(f"   Mean reward: {results.mean_reward:.3f}")
        if results.fairness_index:
            print(f"   Fairness index: {results.fairness_index:.3f}")
        print(f"   Total steps: {results.total_steps}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Performance: {results.total_steps / duration if duration > 0 else 0:.1f} steps/second")

        # Prepare results
        result_data = {
            "config": config.to_dict(),
            "results": {
                "mean_reward": results.mean_reward,
                "fairness_index": results.fairness_index,
                "total_steps": results.total_steps,
                "episode_rewards": {str(k): v for k, v in results.episode_rewards.items()} if hasattr(results, 'episode_rewards') else {},
                "episode_length": getattr(results, 'episode_length', config.episode_length),
                "agent_stats": getattr(results, 'agent_stats', {}),
                "environment_stats": getattr(results, 'environment_stats', {}),
                "emergent_patterns": getattr(results, 'emergent_patterns', {}),
            },
            "timing": {
                "duration": duration,
                "steps_per_second": results.total_steps / duration if duration > 0 else 0,
            }
        }

        # Add telemetry data if available
        if telemetry:
            result_data["telemetry"] = telemetry.get_summary()

        return result_data

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Run benchmark suite."""
    try:
        print("🧪 Running standard benchmarks...")
        
        benchmark = StandardBenchmark()
        
        # Get agent classes
        agent_classes = [AGENT_TYPES[agent_type] for agent_type in args.agent_types]
        
        # Define environments
        environments = args.environments if hasattr(args, 'environments') else ["foraging_sparse", "foraging_dense"]
        
        # Run benchmarks
        results = benchmark.run_all(
            agent_classes=agent_classes,
            environments=environments,
            num_seeds=args.seeds,
            verbose=True
        )
        
        print("\n📈 Benchmark Results:")
        benchmark.print_summary(results)
        
        if args.output:
            benchmark.save_results(results, args.output)
            print(f"💾 Benchmark results saved to {args.output}")
        
        return results

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

def run_distributed(args: argparse.Namespace) -> Dict[str, Any]:
    """Run distributed simulation."""
    try:
        import ray
        
        # Initialize Ray
        if args.ray_address:
            ray.init(address=args.ray_address)
        else:
            ray.init()
        
        print(f"Ray initialized with {ray.cluster_resources()}")
        
        # Load configuration
        if args.config_file:
            config = load_config(args.config_file)
        else:
            config = SwarmConfig(
                num_agents=args.agents,
                episode_length=1000
            )
        
        # Create distributed arena
        arena = DistributedArena(
            config=config,
            num_workers=args.workers
        )
        
        print(f"Running distributed simulation with {args.workers} workers...")
        start_time = time.time()
        
        results = arena.run(episodes=args.episodes)
        
        end_time = time.time()
        duration = end_time - start_time
        
        ray.shutdown()
        
        result_data = {
            "config": config.to_dict(),
            "results": results,
            "timing": {
                "duration": duration,
                "workers": args.workers,
            }
        }
        
        print(f"Distributed simulation completed in {duration:.2f} seconds")
        return result_data

    except ImportError:
        logger.error("Ray is required for distributed execution. Install with: pip install ray")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Distributed simulation failed: {e}")
        sys.exit(1)

def run_scaling_test(args: argparse.Namespace) -> None:
    """Run scaling performance tests."""
    print("📈 Running scaling tests...")
    
    scaling_bench = ScalingBenchmark()
    
    if args.test_type == "weak":
        results = scaling_bench.weak_scaling_test(
            base_agents=args.base_agents,
            base_arena=args.base_arena,
            scale_factors=args.scale_factors,
            metric="throughput"
        )
    else:
        results = scaling_bench.strong_scaling_test(
            num_agents=args.agents,
            num_workers=args.workers,
            metric="episode_time"
        )
    
    print(f"\n📊 Scaling Test Results ({args.test_type} scaling):")
    for scale, result in results.items():
        print(f"   Scale {scale}: {result:.3f}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Scaling results saved to {args.output}")

def create_example_config(args: argparse.Namespace) -> None:
    """Create example configuration file."""
    config = {
        "num_agents": 100,
        "arena_size": [1000, 1000],
        "episode_length": 1000,
        "max_agent_speed": 5.0,
        "observation_radius": 50.0,
        "collision_detection": True,
        "collision_radius": 2.0,
        "seed": 42,
        "reward_config": {
            "resource_collection": 1.0,
            "time_penalty": -0.001,
            "survival_bonus": 0.01
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📝 Example configuration saved to {args.output}")

def show_info(args: argparse.Namespace) -> None:
    """Show system information."""
    if args.agents:
        print("Available agent types:")
        for name, agent_class in AGENT_TYPES.items():
            print(f"  {name}: {agent_class.__doc__ or 'No description'}")
    
    if args.environments:
        print("Available environments:")
        print("  foraging: Resource collection environment")
        print("  foraging_sparse: Sparse resource distribution")
        print("  foraging_dense: Dense resource distribution")
        print("  custom: User-defined custom environments")
    
    if not (args.agents or args.environments):
        print("Swarm Arena System Information")
        print(f"Python version: {sys.version}")
        print(f"NumPy version: {np.__version__}")
        try:
            import ray
            print(f"Ray version: {ray.__version__}")
        except ImportError:
            print("Ray: Not installed")

def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "run":
            results = run_simulation(args)
            if args.output:
                save_results(results, args.output)
        
        elif args.command == "simulate":
            results = run_simulation(args)
            if args.output:
                save_results(results, args.output)
        
        elif args.command == "benchmark":
            results = run_benchmark(args)
            if args.output:
                save_results(results, args.output)
        
        elif args.command == "distributed":
            results = run_distributed(args)
            if hasattr(args, 'output') and args.output:
                save_results(results, args.output)
        
        elif args.command == "scaling":
            run_scaling_test(args)
        
        elif args.command == "config":
            create_example_config(args)
        
        elif args.command == "info":
            show_info(args)
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```
