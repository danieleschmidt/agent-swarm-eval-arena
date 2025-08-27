#!/usr/bin/env python3
"""
Simple demonstration of swarm arena functionality without external dependencies.
Tests core functionality with minimal viable implementation.
"""

import sys
import os
import random
import math
import time
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MockNumpy:
    """Mock numpy for basic functionality when numpy not available."""
    
    @staticmethod
    def mean(arr):
        return sum(arr) / len(arr) if arr else 0
    
    @staticmethod
    def std(arr):
        if not arr:
            return 0
        mean_val = MockNumpy.mean(arr)
        return math.sqrt(sum((x - mean_val) ** 2 for x in arr) / len(arr))
    
    @staticmethod
    def random():
        return random.random()
    
    @staticmethod
    def array(data):
        return list(data)


class SimpleAgent:
    """Simplified agent for demonstration."""
    
    def __init__(self, agent_id: int, x: float = 0, y: float = 0):
        self.id = agent_id
        self.x = x
        self.y = y
        self.energy = 100.0
        self.rewards = []
    
    def act(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """Simple random movement."""
        return {
            'dx': (random.random() - 0.5) * 10,
            'dy': (random.random() - 0.5) * 10
        }
    
    def step(self, action: Dict[str, float]) -> None:
        """Update agent state."""
        self.x += action.get('dx', 0)
        self.y += action.get('dy', 0)
        self.energy -= 1.0


class SimpleArena:
    """Simplified arena implementation."""
    
    def __init__(self, num_agents: int = 10, arena_size: Tuple[int, int] = (1000, 1000)):
        self.num_agents = num_agents
        self.width, self.height = arena_size
        self.agents = {}
        self.step_count = 0
        
        # Initialize agents
        for i in range(num_agents):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            self.agents[i] = SimpleAgent(i, x, y)
    
    def get_observation(self, agent_id: int) -> Dict[str, Any]:
        """Get observation for agent."""
        agent = self.agents[agent_id]
        nearby_agents = []
        
        for other_id, other_agent in self.agents.items():
            if other_id != agent_id:
                distance = math.sqrt((agent.x - other_agent.x)**2 + (agent.y - other_agent.y)**2)
                if distance < 100:  # Vision range
                    nearby_agents.append({
                        'id': other_id,
                        'x': other_agent.x,
                        'y': other_agent.y,
                        'distance': distance
                    })
        
        return {
            'position': (agent.x, agent.y),
            'energy': agent.energy,
            'nearby_agents': nearby_agents,
            'arena_bounds': (self.width, self.height)
        }
    
    def step(self) -> Dict[str, Any]:
        """Execute one simulation step."""
        actions = {}
        observations = {}
        
        # Get actions from all agents
        for agent_id in self.agents:
            obs = self.get_observation(agent_id)
            observations[agent_id] = obs
            actions[agent_id] = self.agents[agent_id].act(obs)
        
        # Execute actions
        for agent_id, action in actions.items():
            self.agents[agent_id].step(action)
            
            # Boundary checking
            agent = self.agents[agent_id]
            agent.x = max(0, min(self.width, agent.x))
            agent.y = max(0, min(self.height, agent.y))
        
        self.step_count += 1
        
        return {
            'step': self.step_count,
            'agents': {aid: {'x': agent.x, 'y': agent.y, 'energy': agent.energy} 
                      for aid, agent in self.agents.items()}
        }
    
    def run_episode(self, max_steps: int = 1000) -> Dict[str, Any]:
        """Run complete episode."""
        episode_data = []
        
        for _ in range(max_steps):
            step_data = self.step()
            episode_data.append(step_data)
            
            # Check termination conditions
            if all(agent.energy <= 0 for agent in self.agents.values()):
                break
        
        # Calculate results
        total_energy = sum(agent.energy for agent in self.agents.values())
        mean_energy = total_energy / len(self.agents)
        
        return {
            'episode_length': len(episode_data),
            'total_steps': self.step_count,
            'final_energy': total_energy,
            'mean_energy': mean_energy,
            'agents_alive': sum(1 for agent in self.agents.values() if agent.energy > 0),
            'episode_data': episode_data[-10:]  # Keep last 10 steps
        }


def main():
    """Run simple arena demonstration."""
    print("🏟️  Swarm Arena - Simple Demo")
    print("=" * 50)
    
    # Test basic functionality
    print("Initializing arena with 20 agents...")
    arena = SimpleArena(num_agents=20, arena_size=(1000, 1000))
    
    print(f"Arena created with {arena.num_agents} agents")
    print(f"Arena size: {arena.width} x {arena.height}")
    
    # Run episode
    print("\nRunning simulation episode...")
    start_time = time.time()
    results = arena.run_episode(max_steps=500)
    end_time = time.time()
    
    # Display results
    print("\n📊 Results:")
    print(f"Episode length: {results['episode_length']} steps")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Steps per second: {results['episode_length'] / (end_time - start_time):.1f}")
    print(f"Final mean energy: {results['mean_energy']:.1f}")
    print(f"Agents surviving: {results['agents_alive']}/{arena.num_agents}")
    
    # Test scaling
    print("\n🚀 Scaling Test:")
    for num_agents in [10, 50, 100]:
        arena = SimpleArena(num_agents=num_agents, arena_size=(2000, 2000))
        start_time = time.time()
        arena.run_episode(max_steps=100)
        end_time = time.time()
        print(f"{num_agents:3d} agents: {end_time - start_time:.3f}s")
    
    print("\n✅ Simple demo completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)