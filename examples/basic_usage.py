#!/usr/bin/env python3
"""
Basic usage example for the Swarm Arena.

This example demonstrates:
1. Creating a basic arena with cooperative and competitive agents
2. Running a simulation
3. Analyzing results
"""

import numpy as np
from swarm_arena import Arena, SwarmConfig, Agent
from swarm_arena.core.agent import CooperativeAgent, CompetitiveAgent


def main():
    """Run basic swarm arena example."""
    print("🏟️ Swarm Arena Basic Usage Example")
    print("=" * 50)
    
    # Configure arena
    config = SwarmConfig(
        num_agents=50,
        arena_size=(800, 600), 
        episode_length=500,
        resource_spawn_rate=0.05,
        seed=42
    )
    
    print(f"Arena size: {config.arena_size}")
    print(f"Episode length: {config.episode_length} steps")
    print(f"Resource spawn rate: {config.resource_spawn_rate}")
    
    # Create arena
    arena = Arena(config)
    
    # Add mixed agent types
    print(f"\nAdding agents:")
    arena.add_agents(CooperativeAgent, count=20)
    print(f"  - 20 cooperative agents")
    
    arena.add_agents(CompetitiveAgent, count=20) 
    print(f"  - 20 competitive agents")
    
    arena.add_agents(Agent, count=10)
    print(f"  - 10 default agents")
    
    print(f"Total agents: {len(arena.agents)}")
    
    # Run simulation
    print(f"\n🚀 Running simulation...")
    results = arena.run(episodes=3, verbose=True)
    
    # Analyze results
    print(f"\n📊 Results Analysis:")
    print(f"Mean reward per episode: {results.mean_reward:.3f}")
    
    if results.fairness_index is not None:
        print(f"Fairness index: {results.fairness_index:.3f}")
    
    print(f"Total simulation steps: {results.total_steps}")
    
    # Agent type performance comparison
    print(f"\n🤖 Agent Performance by Type:")
    
    coop_rewards = []
    comp_rewards = []
    default_rewards = []
    
    for agent_id, stats in results.agent_stats.items():
        total_reward = stats["total_reward"]
        
        if agent_id < 20:  # Cooperative agents (first 20)
            coop_rewards.append(total_reward)
        elif agent_id < 40:  # Competitive agents (next 20) 
            comp_rewards.append(total_reward)
        else:  # Default agents (last 10)
            default_rewards.append(total_reward)
    
    print(f"Cooperative agents - Mean reward: {np.mean(coop_rewards):.3f} ± {np.std(coop_rewards):.3f}")
    print(f"Competitive agents - Mean reward: {np.mean(comp_rewards):.3f} ± {np.std(comp_rewards):.3f}")
    print(f"Default agents - Mean reward: {np.mean(default_rewards):.3f} ± {np.std(default_rewards):.3f}")
    
    # Resource collection stats
    total_resources = sum(stats["resources_collected"] for stats in results.agent_stats.values())
    print(f"\n🎯 Resource Collection:")
    print(f"Total resources collected: {total_resources}")
    print(f"Resources per agent: {total_resources / len(arena.agents):.2f}")
    
    # Environment stats
    env_stats = results.environment_stats
    print(f"\n🌍 Environment Statistics:")
    print(f"Final step: {env_stats.get('step', 0)}")
    print(f"Active resources: {env_stats.get('active_resources', 0)}")
    print(f"Total resources spawned: {env_stats.get('total_resources_spawned', 0)}")
    
    print(f"\n✅ Simulation complete!")


if __name__ == "__main__":
    main()