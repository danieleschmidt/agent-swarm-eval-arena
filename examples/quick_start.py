#!/usr/bin/env python3
"""
Quick Start Example for Swarm Arena
Demonstrates basic usage of the platform with various agent types.
"""

import numpy as np
from swarm_arena import (
    Arena, SwarmConfig, Agent, CooperativeAgent, CompetitiveAgent, 
    RandomAgent, set_global_seed
)

def basic_simulation():
    """Run a basic simulation with mixed agent types."""
    print("=== Basic Swarm Arena Simulation ===")
    
    # Create configuration
    config = SwarmConfig(
        num_agents=50,
        arena_size=(800, 600),
        episode_length=500,
        resource_spawn_rate=0.15,
        seed=42
    )
    
    # Set seed for reproducibility
    set_global_seed(config.seed)
    
    # Create arena
    arena = Arena(config)
    
    # Add mixed agent types
    arena.add_agents(CooperativeAgent, count=20)
    arena.add_agents(CompetitiveAgent, count=20) 
    arena.add_agents(RandomAgent, count=10)
    
    print(f"Created arena with {len(arena.agents)} agents")
    print(f"Arena size: {config.arena_size}")
    print(f"Episode length: {config.episode_length} steps")
    
    # Run simulation
    results = arena.run(episodes=5, verbose=True)
    
    # Display results
    print(f"\n=== Results ===")
    print(f"Mean reward per episode: {results.mean_reward:.3f}")
    print(f"Fairness index: {results.fairness_index:.3f}")
    print(f"Total simulation steps: {results.total_steps}")
    
    # Agent type analysis
    coop_agents = [aid for aid, agent in arena.agents.items() 
                   if isinstance(agent, CooperativeAgent)]
    comp_agents = [aid for aid, agent in arena.agents.items() 
                   if isinstance(agent, CompetitiveAgent)]
    
    coop_rewards = [results.agent_stats[aid]["total_reward"] for aid in coop_agents]
    comp_rewards = [results.agent_stats[aid]["total_reward"] for aid in comp_agents]
    
    print(f"\nCooperative agents average reward: {np.mean(coop_rewards):.3f}")
    print(f"Competitive agents average reward: {np.mean(comp_rewards):.3f}")
    
    return results

def performance_test():
    """Test performance with larger numbers of agents."""
    print("\n=== Performance Test ===")
    
    import time
    
    configs = [
        {"num_agents": 100, "name": "Small"},
        {"num_agents": 500, "name": "Medium"},
        {"num_agents": 1000, "name": "Large"}
    ]
    
    for config_params in configs:
        config = SwarmConfig(
            num_agents=config_params["num_agents"],
            arena_size=(1500, 1500),
            episode_length=200,
            seed=42
        )
        
        arena = Arena(config)
        arena.add_agents(Agent, count=config.num_agents)
        
        print(f"\n{config_params['name']} scale test ({config.num_agents} agents):")
        
        start_time = time.time()
        results = arena.run(episodes=3, verbose=False)
        end_time = time.time()
        
        duration = end_time - start_time
        steps_per_second = results.total_steps / duration
        
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Performance: {steps_per_second:.1f} steps/second")
        print(f"  Mean reward: {results.mean_reward:.3f}")

def agent_comparison():
    """Compare different agent types in the same environment."""
    print("\n=== Agent Type Comparison ===")
    
    agent_types = [
        (Agent, "Basic"),
        (CooperativeAgent, "Cooperative"),
        (CompetitiveAgent, "Competitive"),
        (RandomAgent, "Random")
    ]
    
    results = {}
    
    for agent_class, name in agent_types:
        config = SwarmConfig(
            num_agents=30,
            arena_size=(600, 600),
            episode_length=300,
            seed=42
        )
        
        arena = Arena(config)
        arena.add_agents(agent_class, count=config.num_agents)
        
        print(f"\nTesting {name} agents...")
        sim_results = arena.run(episodes=3, verbose=False)
        
        results[name] = {
            "mean_reward": sim_results.mean_reward,
            "resources_collected": sim_results.environment_stats.get("collected_resources", 0),
            "efficiency": sim_results.environment_stats.get("collected_resources", 0) / config.num_agents
        }
        
        print(f"  Mean reward: {sim_results.mean_reward:.3f}")
        print(f"  Resources collected: {results[name]['resources_collected']}")
        print(f"  Efficiency: {results[name]['efficiency']:.2f} resources/agent")
    
    # Find best performing agent type
    best_agent = max(results.keys(), key=lambda x: results[x]["mean_reward"])
    print(f"\nBest performing agent type: {best_agent}")
    print(f"  Reward: {results[best_agent]['mean_reward']:.3f}")
    print(f"  Efficiency: {results[best_agent]['efficiency']:.2f}")

def interactive_demo():
    """Interactive demonstration of arena capabilities."""
    print("\n=== Interactive Demo ===")
    print("This demo shows real-time statistics during simulation")
    
    config = SwarmConfig(
        num_agents=100,
        arena_size=(1000, 1000),
        episode_length=1000,
        resource_spawn_rate=0.1,
        seed=42
    )
    
    arena = Arena(config)
    
    # Add diverse agent mix
    arena.add_agents(CooperativeAgent, count=30)
    arena.add_agents(CompetitiveAgent, count=30)
    arena.add_agents(Agent, count=40)
    
    print("Starting simulation with real-time monitoring...")
    print("(Watch for periodic updates showing agent activity)")
    
    # Run with detailed monitoring
    results = arena.run(episodes=2, verbose=True)
    
    print(f"\nFinal Statistics:")
    print(f"Total episodes: 2")
    print(f"Mean reward: {results.mean_reward:.3f}")
    print(f"Fairness index: {results.fairness_index:.3f}")
    
    # Show agent diversity stats
    alive_agents = sum(1 for stats in results.agent_stats.values() if stats["alive"])
    print(f"Agents still alive: {alive_agents}/{len(results.agent_stats)}")
    
    avg_resources = np.mean([stats["resources_collected"] 
                           for stats in results.agent_stats.values()])
    print(f"Average resources per agent: {avg_resources:.2f}")

if __name__ == "__main__":
    """Run all demonstrations."""
    try:
        # Basic simulation
        basic_simulation()
        
        # Performance testing
        performance_test()
        
        # Agent comparison
        agent_comparison()
        
        # Interactive demo
        interactive_demo()
        
        print("\n=== Quick Start Complete ===")
        print("You can now:")
        print("1. Modify the examples above")
        print("2. Create custom agent types")
        print("3. Run larger simulations")
        print("4. Use the CLI: swarm-arena --help")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()