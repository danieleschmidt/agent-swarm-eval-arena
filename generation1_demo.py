#!/usr/bin/env python3
"""
Generation 1 Demo: Make It Work (Simple)
Demonstrates basic swarm arena functionality with mixed agent types.
"""

import sys
sys.path.insert(0, '.')

from swarm_arena import Arena, SwarmConfig
from swarm_arena.core.agent import CooperativeAgent, CompetitiveAgent, RandomAgent

def main():
    print("🏟️ SWARM ARENA - GENERATION 1 DEMO")
    print("=" * 50)
    
    # Create configuration
    config = SwarmConfig(
        num_agents=50,
        arena_size=(800, 600),
        episode_length=200,
        resource_spawn_rate=0.15,
        seed=42
    )
    print(f"Arena: {config.arena_size[0]}x{config.arena_size[1]}")
    print(f"Episodes: {config.episode_length} steps")
    print(f"Resource spawn rate: {config.resource_spawn_rate}")
    
    # Create arena
    arena = Arena(config)
    print(f"✅ Arena initialized")
    
    # Add mixed agent population
    arena.add_agents(CooperativeAgent, 20)
    arena.add_agents(CompetitiveAgent, 20) 
    arena.add_agents(RandomAgent, 10)
    
    print(f"👥 Added {len(arena.agents)} agents:")
    print(f"   • 20 Cooperative agents")
    print(f"   • 20 Competitive agents") 
    print(f"   • 10 Random agents")
    
    # Run simulation
    print(f"\n🚀 Running simulation...")
    results = arena.run(episodes=3, verbose=True)
    
    # Display results
    print(f"\n📊 RESULTS")
    print(f"-" * 30)
    print(f"Mean reward: {results.mean_reward:.3f}")
    if results.fairness_index:
        print(f"Fairness index: {results.fairness_index:.3f}")
    print(f"Total steps: {results.total_steps}")
    print(f"Episode length: {results.episode_length}")
    
    # Agent performance breakdown
    print(f"\n🎯 AGENT PERFORMANCE")
    print(f"-" * 30)
    
    coop_rewards = []
    comp_rewards = []
    rand_rewards = []
    
    for agent_id, rewards in results.episode_rewards.items():
        total_reward = sum(rewards)
        if agent_id < 20:  # Cooperative agents
            coop_rewards.append(total_reward)
        elif agent_id < 40:  # Competitive agents
            comp_rewards.append(total_reward)
        else:  # Random agents
            rand_rewards.append(total_reward)
    
    if coop_rewards:
        print(f"Cooperative agents: {sum(coop_rewards)/len(coop_rewards):.3f} avg reward")
    if comp_rewards:
        print(f"Competitive agents: {sum(comp_rewards)/len(comp_rewards):.3f} avg reward")
    if rand_rewards:
        print(f"Random agents: {sum(rand_rewards)/len(rand_rewards):.3f} avg reward")
    
    # Environment stats
    env_stats = results.environment_stats
    if 'collected_resources' in env_stats:
        print(f"\n🌟 ENVIRONMENT")
        print(f"-" * 30)
        print(f"Resources collected: {env_stats['collected_resources']}")
        print(f"Collection efficiency: {env_stats.get('efficiency', 0.0):.3f}")
    
    print(f"\n✅ Generation 1 Demo Complete!")
    print(f"🎉 Basic functionality working perfectly!")

if __name__ == "__main__":
    main()