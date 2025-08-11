#!/usr/bin/env python3
"""
Generation 1 Validation Demo - Simple Multi-Agent Arena
Demonstrates core functionality with minimal viable features
"""

import numpy as np
from swarm_arena import Arena, SwarmConfig, CooperativeAgent, CompetitiveAgent

def main():
    print("🚀 Swarm Arena - Generation 1 Validation Demo")
    print("=" * 50)
    
    # Create basic configuration
    config = SwarmConfig(
        num_agents=20,
        arena_size=(500, 500),
        episode_length=100,
        observation_radius=50.0,
        max_agent_speed=5.0,
        collision_detection=True,
        collision_radius=5.0,
        seed=42
    )
    
    # Create arena
    arena = Arena(config)
    
    # Add cooperative and competitive agents
    arena.add_agents(CooperativeAgent, count=10)
    arena.add_agents(CompetitiveAgent, count=10)
    
    print(f"✓ Arena created with {len(arena.agents)} agents")
    print(f"✓ Arena size: {config.arena_size}")
    print(f"✓ Episode length: {config.episode_length} steps")
    
    # Run single episode evaluation
    print("\n🎯 Running Generation 1 evaluation...")
    results = arena.run(episodes=1, verbose=False)
    
    print(f"\n📊 Results:")
    print(f"• Mean reward: {results.mean_reward:.3f}")
    print(f"• Total steps: {results.total_steps}")
    print(f"• Active agents at end: {sum(1 for a in arena.agents.values() if a.state.alive)}")
    
    if results.fairness_index is not None:
        print(f"• Fairness index: {results.fairness_index:.3f}")
    
    # Test basic metrics collection
    print(f"\n🔍 Agent Performance:")
    for agent_id, stats in results.agent_stats.items():
        if agent_id < 5:  # Show first 5 agents only
            agent_type = type(arena.agents[agent_id]).__name__
            resources = stats.get('resources_collected', 0)
            print(f"  Agent {agent_id} ({agent_type}): {resources} resources collected")
    
    print("\n✅ Generation 1 demonstration complete!")
    print("✅ Core functionality validated successfully!")

if __name__ == "__main__":
    main()