"""CLI interface for swarm-arena."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

from .core.arena import Arena
from .core.config import SwarmConfig
from .core.agent import CooperativeAgent, CompetitiveAgent, RandomAgent
from .benchmarks.standard import StandardBenchmark
from .benchmarks.scaling import ScalingBenchmark


def create_arena_from_config(config_path: str) -> Arena:
    """Create arena from configuration file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = SwarmConfig(**config_dict)
    return Arena(config)


def run_simulation(args) -> None:
    """Run a basic simulation."""
    config = SwarmConfig(
        num_agents=args.agents,
        arena_size=(args.width, args.height),
        episode_length=args.steps,
        seed=args.seed
    )
    
    arena = Arena(config)
    
    # Add agents based on type
    if args.agent_type == "cooperative":
        arena.add_agents(CooperativeAgent, count=args.agents)
    elif args.agent_type == "competitive":
        arena.add_agents(CompetitiveAgent, count=args.agents)
    elif args.agent_type == "mixed":
        arena.add_agents(CooperativeAgent, count=args.agents // 2)
        arena.add_agents(CompetitiveAgent, count=args.agents - args.agents // 2)
    else:
        arena.add_agents(RandomAgent, count=args.agents)
    
    print(f"🏟️  Starting simulation with {args.agents} agents...")
    print(f"📏 Arena size: {args.width}x{args.height}")
    print(f"⏱️  Episode length: {args.steps} steps")
    
    results = arena.run(episodes=args.episodes, verbose=args.verbose)
    
    print(f"\n📊 Simulation Results:")
    print(f"   Mean reward: {results.mean_reward:.3f}")
    if results.fairness_index:
        print(f"   Fairness index: {results.fairness_index:.3f}")
    print(f"   Total steps: {results.total_steps}")
    
    if args.output:
        output_data = {
            "config": config.__dict__,
            "results": {
                "mean_reward": results.mean_reward,
                "fairness_index": results.fairness_index,
                "total_steps": results.total_steps,
                "episode_rewards": {str(k): v for k, v in results.episode_rewards.items()}
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"💾 Results saved to {args.output}")


def run_benchmark(args) -> None:
    """Run standard benchmarks."""
    print("🧪 Running standard benchmarks...")
    
    benchmark = StandardBenchmark()
    
    # Define agent classes to test
    agent_classes = [CooperativeAgent, CompetitiveAgent, RandomAgent]
    
    # Define environments
    environments = ["foraging_sparse", "foraging_dense"]
    
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


def run_scaling_test(args) -> None:
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


def create_example_config(args) -> None:
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


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🏟️  Swarm Arena - Multi-Agent Reinforcement Learning Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic simulation
  swarm-arena simulate --agents 100 --episodes 5

  # Run with mixed agent types
  swarm-arena simulate --agents 200 --agent-type mixed --verbose

  # Run benchmarks
  swarm-arena benchmark --seeds 10 --output results.json

  # Run scaling tests
  swarm-arena scaling --test-type weak --scale-factors 1 2 4 8

  # Create example configuration
  swarm-arena config --output example_config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Simulate command
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
    sim_parser.set_defaults(func=run_simulation)
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    bench_parser.add_argument("--output", type=str, help="Output file for results")
    bench_parser.set_defaults(func=run_benchmark)
    
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
    scale_parser.set_defaults(func=run_scaling_test)
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Create example configuration")
    config_parser.add_argument("--output", type=str, default="config.json",
                             help="Output configuration file")
    config_parser.set_defaults(func=create_example_config)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()