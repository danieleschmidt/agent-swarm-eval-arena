#!/usr/bin/env python3
"""
Generation 1 Complete Demo - Showcase all implemented functionality
"""

import sys
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from swarm_arena import Arena, SwarmConfig, CooperativeAgent, CompetitiveAgent
from swarm_arena.research import EmergenceDetector, FairnessAnalyzer, ExperimentLogger
from swarm_arena.research.communication import MessageChannel, NegotiationProtocol
from swarm_arena.cli import main as cli_main


def demo_basic_simulation():
    """Demonstrate basic swarm simulation functionality."""
    print("🏟️  BASIC SIMULATION DEMO")
    print("=" * 50)
    
    # Create configuration
    config = SwarmConfig(
        num_agents=50,
        arena_size=(500, 500),
        episode_length=100,
        seed=42
    )
    
    # Create arena
    arena = Arena(config)
    
    # Add mixed agent types
    arena.add_agents(CooperativeAgent, count=25)
    arena.add_agents(CompetitiveAgent, count=25)
    
    print(f"Created arena with {len(arena.agents)} agents")
    print(f"Arena size: {config.arena_size}")
    
    # Run simulation
    results = arena.run(episodes=2, verbose=True)
    
    print(f"\n📊 Results:")
    print(f"   Mean reward: {results.mean_reward:.3f}")
    print(f"   Fairness index: {results.fairness_index:.3f}")
    print(f"   Total steps: {results.total_steps}")
    
    return results


def demo_research_capabilities():
    """Demonstrate research and analysis capabilities."""
    print("\n🔬 RESEARCH CAPABILITIES DEMO")
    print("=" * 50)
    
    # Create sample trajectory data
    print("Creating sample trajectory data...")
    trajectories = {}
    
    # Generate realistic swarm trajectories
    for agent_id in range(20):
        trajectory = []
        x, y = np.random.uniform(50, 450, 2)  # Random start position
        
        for t in range(50):
            # Add some flocking behavior
            if t > 10:
                # Move towards center with some noise
                center_x, center_y = 250, 250
                dx = (center_x - x) * 0.1 + np.random.normal(0, 5)
                dy = (center_y - y) * 0.1 + np.random.normal(0, 5)
                x += dx
                y += dy
            else:
                # Random walk initially
                x += np.random.normal(0, 10)
                y += np.random.normal(0, 10)
            
            # Keep in bounds
            x = np.clip(x, 0, 500)
            y = np.clip(y, 0, 500)
            
            trajectory.append((x, y))
        
        trajectories[agent_id] = trajectory
    
    # Emergence detection
    print("\n🌊 Analyzing emergent patterns...")
    detector = EmergenceDetector()
    patterns = detector.analyze(trajectories)
    
    print(f"Detected {len(patterns)} emergent patterns:")
    for pattern in patterns:
        print(f"  • {pattern.name}: confidence={pattern.confidence:.3f}, "
              f"duration={pattern.duration}, agents={len(pattern.agents)}")
    
    # Fairness analysis
    print("\n⚖️  Analyzing fairness...")
    fairness_analyzer = FairnessAnalyzer()
    
    # Sample allocations
    allocations = {i: np.random.exponential(10) for i in range(20)}
    metrics = fairness_analyzer.analyze_allocation(allocations)
    
    print(f"Fairness metrics:")
    print(f"  • Gini coefficient: {metrics.gini_coefficient:.3f}")
    print(f"  • Envy-freeness: {metrics.envy_freeness:.3f}")
    print(f"  • Jain's fairness: {metrics.jain_fairness_index:.3f}")
    
    return patterns, metrics


def demo_communication():
    """Demonstrate communication protocols."""
    print("\n📡 COMMUNICATION DEMO")
    print("=" * 50)
    
    # Create message channel
    channel = MessageChannel(
        max_range=100.0,
        bandwidth_limit=5,
        noise_probability=0.1
    )
    
    # Create negotiation protocol
    protocol = NegotiationProtocol()
    
    # Agent positions for range checking
    agent_positions = {
        0: np.array([100, 100]),
        1: np.array([150, 150]),  # Within range
        2: np.array([300, 300])   # Out of range
    }
    
    # Agent 0 sends a proposal to Agent 1
    proposal_data = {
        "sender_id": 0,
        "receiver_id": 1,
        "resource_split": [0.6, 0.4]
    }
    
    proposal_message = protocol.encode_proposal(proposal_data)
    success = channel.send_message(proposal_message, agent_positions, current_step=1)
    
    print(f"Message sent: {success}")
    
    # Agent 1 receives messages
    messages = channel.get_messages_for_agent(1, current_step=1)
    print(f"Agent 1 received {len(messages)} messages")
    
    for msg in messages:
        response = protocol.decode_response(msg)
        print(f"  • Message type: {msg.message_type.value}")
        print(f"  • Content: {response}")
    
    # Show communication stats
    stats = channel.get_statistics()
    print(f"\nCommunication statistics: {stats}")
    
    return channel, protocol


def demo_experiment_logging():
    """Demonstrate experiment logging and reproducibility."""
    print("\n📝 EXPERIMENT LOGGING DEMO")
    print("=" * 50)
    
    # Create experiment logger
    logger = ExperimentLogger(base_path="demo_experiments")
    
    # Start experiment
    experiment = logger.start_experiment(
        name="Generation 1 Demo",
        description="Demonstrating basic swarm arena functionality",
        parameters={"num_agents": 50, "episode_length": 100},
        seed=42
    )
    
    # Log some metrics
    logger.log_metric("demo_metric", 0.85)
    logger.log_metrics({
        "accuracy": 0.92,
        "efficiency": 0.78,
        "fairness": 0.65
    })
    
    # Finish experiment
    saved_path = logger.finish_experiment(success=True)
    print(f"Experiment saved to: {saved_path}")
    
    return logger, experiment


def demo_cli_functionality():
    """Demonstrate CLI functionality."""
    print("\n💻 CLI DEMO")
    print("=" * 50)
    
    print("Available CLI commands:")
    print("  • swarm-arena simulate --agents 100 --episodes 5")
    print("  • swarm-arena benchmark --seeds 5")
    print("  • swarm-arena scaling --test-type weak")
    print("  • swarm-arena config --output example_config.json")
    
    print("\nFor full CLI demo, run:")
    print("  source .venv/bin/activate && swarm-arena --help")


def main():
    """Run complete Generation 1 demonstration."""
    print("🚀 SWARM ARENA - GENERATION 1 COMPLETE DEMO")
    print("🤖 Autonomous SDLC Execution - Making It Work!")
    print("=" * 60)
    
    try:
        # Basic simulation
        sim_results = demo_basic_simulation()
        
        # Research capabilities
        patterns, fairness_metrics = demo_research_capabilities()
        
        # Communication
        channel, protocol = demo_communication()
        
        # Experiment logging
        logger, experiment = demo_experiment_logging()
        
        # CLI functionality
        demo_cli_functionality()
        
        print("\n🎉 GENERATION 1 COMPLETE!")
        print("=" * 60)
        print("✅ Basic simulation functionality")
        print("✅ Research tools (emergence, fairness)")
        print("✅ Communication protocols")
        print("✅ Experiment logging & reproducibility")
        print("✅ Command-line interface")
        
        print(f"\n📈 Demo Summary:")
        print(f"   • Simulated {len(sim_results.episode_rewards)} agents")
        print(f"   • Detected {len(patterns)} emergent patterns")
        print(f"   • Fairness index: {fairness_metrics.gini_coefficient:.3f}")
        print(f"   • Communication success rate: {channel.get_statistics()}")
        
        print(f"\n🎯 Ready for Generation 2: Adding robustness and reliability!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)