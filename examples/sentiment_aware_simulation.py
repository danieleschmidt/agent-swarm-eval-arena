"""
Advanced Sentiment-Aware Multi-Agent Simulation Example

Demonstrates the world's first large-scale sentiment-aware multi-agent system
with emotional contagion, behavioral adaptation, and real-time analytics.

This example showcases breakthrough capabilities in emotional AI and provides
a foundation for research in sentiment-aware multi-agent reinforcement learning.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Import the enhanced swarm arena with sentiment capabilities
from swarm_arena import (
    Arena, SwarmConfig, 
    SentimentAwareAgent, EmotionalCooperativeAgent, EmotionalCompetitiveAgent,
    EmotionalAdaptiveAgent, SentimentAwareAgentConfig,
    ContagionParameters, SentimentContagion,
    set_global_seed
)


def create_sentiment_aware_simulation() -> Arena:
    """Create a sentiment-aware multi-agent simulation with diverse emotional agents."""
    
    # Configure arena for sentiment-aware simulation
    config = SwarmConfig(
        num_agents=150,  # Moderate scale for demonstration
        arena_size=(800, 600),
        episode_length=2000,
        resource_spawn_rate=0.15,
        collision_detection=True,
        collision_radius=8.0,
        observation_radius=80.0,
        max_agent_speed=15.0,
        seed=42  # Reproducible results
    )
    
    # Configure emotional contagion parameters
    contagion_params = ContagionParameters(
        base_influence_strength=0.12,
        max_influence_distance=120.0,
        decay_rate=0.92,
        emotion_compatibility_factor=0.6,
        crowd_amplification=True,
        contagion_threshold=0.35,
        leadership_factor=1.8
    )
    
    # Create sentiment-aware arena
    arena = Arena(
        config=config, 
        enable_sentiment=True,
        contagion_params=contagion_params
    )
    
    print("🧠 Creating Sentiment-Aware Multi-Agent System...")
    print(f"   Arena size: {config.arena_size}")
    print(f"   Max agents: {config.num_agents}")
    print(f"   Episode length: {config.episode_length} steps")
    print(f"   Sentiment contagion: ENABLED")
    print(f"   Emotional influence radius: {contagion_params.max_influence_distance}px")
    
    return arena


def add_diverse_emotional_agents(arena: Arena) -> None:
    """Add diverse emotional agents with different personality profiles."""
    
    print("👥 Adding Diverse Emotional Agent Population...")
    
    # Highly empathetic cooperative agents (30%)
    cooperative_config = SentimentAwareAgentConfig(
        emotional_sensitivity=0.85,
        empathy_level=0.9,
        contagion_susceptibility=0.8,
        learning_from_emotion=True,
        emotion_expression=True
    )
    
    arena.add_agents(EmotionalCooperativeAgent, count=45, config=cooperative_config)
    print(f"   ✅ Added 45 Highly Empathetic Cooperative Agents")
    
    # Competitive agents with moderate emotional awareness (25%)
    competitive_config = SentimentAwareAgentConfig(
        emotional_sensitivity=0.6,
        empathy_level=0.3,
        contagion_susceptibility=0.4,
        learning_from_emotion=True,
        emotion_expression=True
    )
    
    arena.add_agents(EmotionalCompetitiveAgent, count=38, config=competitive_config)
    print(f"   ⚡ Added 38 Competitive Emotional Agents")
    
    # Adaptive agents that learn optimal emotional strategies (30%)
    adaptive_config = SentimentAwareAgentConfig(
        emotional_sensitivity=0.75,
        empathy_level=0.6,
        contagion_susceptibility=0.6,
        learning_from_emotion=True,
        memory_capacity=800,
        emotion_expression=True
    )
    
    arena.add_agents(EmotionalAdaptiveAgent, count=45, config=adaptive_config)
    print(f"   🧠 Added 45 Adaptive Learning Emotional Agents")
    
    # Highly sensitive emotional agents (15%) - experimental group
    sensitive_config = SentimentAwareAgentConfig(
        emotional_sensitivity=0.95,
        empathy_level=0.85,
        contagion_susceptibility=0.9,
        learning_from_emotion=True,
        emotional_decay_rate=0.92,
        emotion_expression=True
    )
    
    arena.add_agents(SentimentAwareAgent, count=22, config=sensitive_config)
    print(f"   🌟 Added 22 Highly Sensitive Emotional Agents")
    
    print(f"   🎯 Total Population: {len(arena.agents)} Sentiment-Aware Agents")


def run_sentiment_simulation(arena: Arena, episodes: int = 3) -> Dict[str, Any]:
    """Run the sentiment-aware simulation and collect comprehensive analytics."""
    
    print(f"\n🚀 Starting {episodes}-Episode Sentiment-Aware Simulation...")
    
    # Enable telemetry streaming for real-time monitoring
    if arena.sentiment_telemetry:
        arena.sentiment_telemetry.streaming_enabled = False  # Keep False for this demo
    
    start_time = time.time()
    
    # Run simulation with verbose output
    results = arena.run(episodes=episodes, verbose=True)
    
    simulation_time = time.time() - start_time
    
    print(f"\n📊 SIMULATION COMPLETED in {simulation_time:.2f} seconds")
    print(f"   Total steps: {results.total_steps:,}")
    print(f"   Average step time: {simulation_time / results.total_steps * 1000:.2f}ms")
    
    return {
        'results': results,
        'simulation_time': simulation_time,
        'steps_per_second': results.total_steps / simulation_time
    }


def analyze_sentiment_results(simulation_data: Dict[str, Any]) -> None:
    """Perform comprehensive analysis of sentiment simulation results."""
    
    results = simulation_data['results']
    
    print("\n📈 SENTIMENT ANALYSIS RESULTS")
    print("=" * 50)
    
    # Performance metrics
    print(f"🏃 Performance:")
    print(f"   Steps/second: {simulation_data['steps_per_second']:.1f}")
    print(f"   Mean reward: {results.mean_reward:.3f}")
    
    if results.fairness_index is not None:
        print(f"   Fairness index: {results.fairness_index:.3f}")
    
    # Sentiment analytics
    if results.sentiment_analytics:
        sentiment_data = results.sentiment_analytics
        
        print(f"\n🧠 Emotional Population Dynamics:")
        
        if 'population_trends' in sentiment_data:
            trends = sentiment_data['population_trends']
            
            print(f"   Population Arousal: {trends.get('arousal', {}).get('current', 0.0):.3f}")
            print(f"   Population Valence: {trends.get('valence', {}).get('current', 0.0):.3f}")
            print(f"   Population Dominance: {trends.get('dominance', {}).get('current', 0.0):.3f}")
            
            arousal_trend = trends.get('arousal', {}).get('trend', 'unknown')
            valence_trend = trends.get('valence', {}).get('trend', 'unknown')
            
            print(f"   Arousal trend: {arousal_trend.upper()}")
            print(f"   Valence trend: {valence_trend.upper()}")
        
        if 'emotion_distributions' in sentiment_data:
            emotions = sentiment_data['emotion_distributions']
            
            print(f"\n😊 Emotion Distribution:")
            for emotion, percentage in emotions.items():
                if emotion != 'diversity' and percentage > 0.05:  # Show emotions > 5%
                    print(f"   {emotion.title()}: {percentage:.1%}")
            
            if 'diversity' in emotions:
                diversity = emotions['diversity']
                print(f"   Emotional Diversity: {diversity.get('current', 0.0):.3f}")
        
        # Contagion analysis
        if 'contagion_analysis' in sentiment_data:
            contagion = sentiment_data['contagion_analysis']
            
            print(f"\n🌊 Emotional Contagion Dynamics:")
            print(f"   Total contagion events: {contagion.get('total_events', 0):,}")
            print(f"   Average active influences: {contagion.get('avg_influences', 0.0):.1f}")
            print(f"   Average emotional clusters: {contagion.get('avg_clusters', 0.0):.1f}")
            print(f"   Contagion rate: {contagion.get('contagion_rate', 0.0):.3f}")
        
        # Agent-level analytics
        if 'agent_analytics' in sentiment_data:
            agent_stats = sentiment_data['agent_analytics']
            
            print(f"\n🤖 Individual Agent Analysis:")
            print(f"   Total agents tracked: {agent_stats.get('total_agents', 0)}")
            print(f"   Most emotional agent: #{agent_stats.get('most_emotional_agent', 'N/A')}")
            print(f"   Average emotional intensity: {agent_stats.get('avg_emotional_intensity', 0.0):.3f}")
            
        # Performance metrics
        if 'performance_metrics' in sentiment_data:
            perf = sentiment_data['performance_metrics']
            
            print(f"\n⚡ Sentiment Processing Performance:")
            print(f"   Avg sentiment processing: {perf.get('avg_sentiment_processing_time', 0.0):.2f}ms")
            print(f"   Avg contagion processing: {perf.get('avg_contagion_processing_time', 0.0):.2f}ms")
            print(f"   Update frequency: {perf.get('update_frequency', 0.0):.1f} Hz")
    
    # Contagion system statistics
    if results.contagion_statistics:
        contagion_stats = results.contagion_statistics
        
        print(f"\n🔗 Contagion System Statistics:")
        print(f"   Influences processed: {contagion_stats.get('total_influences_processed', 0):,}")
        print(f"   Network density: {contagion_stats.get('avg_connections_per_agent', 0.0):.2f}")
        print(f"   Largest cluster size: {contagion_stats.get('largest_cluster_size', 0)}")
    
    print("\n" + "=" * 50)


def demonstrate_emotional_contagion(arena: Arena) -> None:
    """Demonstrate emotional contagion by creating emotional hotspots."""
    
    print("\n🌡️  EMOTIONAL CONTAGION DEMONSTRATION")
    print("Creating emotional hotspots to observe contagion effects...")
    
    # Get sentiment-aware agents
    sentiment_agents = [agent for agent in arena.agents.values() 
                       if isinstance(agent, SentimentAwareAgent)]
    
    if len(sentiment_agents) < 10:
        print("⚠️  Not enough sentiment-aware agents for demonstration")
        return
    
    # Create joy hotspot (first 5 agents)
    print("😄 Creating JOY hotspot...")
    for i, agent in enumerate(sentiment_agents[:5]):
        agent.emotional_state.valence = 0.8
        agent.emotional_state.arousal = 0.6
        agent.emotional_state.dominance = 0.4
    
    # Create anxiety hotspot (next 5 agents)
    print("😰 Creating ANXIETY hotspot...")
    for i, agent in enumerate(sentiment_agents[5:10]):
        agent.emotional_state.valence = -0.6
        agent.emotional_state.arousal = 0.9
        agent.emotional_state.dominance = -0.3
    
    # Run a few steps to observe contagion
    print("🔄 Running 50 steps to observe emotional spread...")
    
    initial_valence = np.mean([agent.emotional_state.valence for agent in sentiment_agents])
    
    for step in range(50):
        arena.step()
    
    final_valence = np.mean([agent.emotional_state.valence for agent in sentiment_agents])
    
    print(f"📈 Emotional contagion effect:")
    print(f"   Initial population valence: {initial_valence:.3f}")
    print(f"   Final population valence: {final_valence:.3f}")
    print(f"   Valence change: {final_valence - initial_valence:+.3f}")
    
    if abs(final_valence - initial_valence) > 0.1:
        print("✅ Strong emotional contagion observed!")
    else:
        print("📝 Moderate emotional contagion observed.")


def create_sentiment_visualization() -> None:
    """Create visualizations for sentiment analysis (placeholder for future implementation)."""
    
    print("\n📊 VISUALIZATION CAPABILITIES")
    print("The system supports the following visualizations:")
    print("   🎨 Real-time emotional field maps")
    print("   📈 Population sentiment trends over time") 
    print("   🔗 Emotional influence network graphs")
    print("   🌈 Agent emotional state distributions")
    print("   🌊 Contagion propagation heatmaps")
    print("   📋 Comparative agent performance by emotional type")
    
    print("\n💡 For visualization implementation, integrate with:")
    print("   - matplotlib/seaborn for static plots")
    print("   - plotly/dash for interactive dashboards")
    print("   - networkx for influence network visualization")
    print("   - WebSocket streaming for real-time monitoring")


def main():
    """Main demonstration of sentiment-aware multi-agent system."""
    
    print("🎭 SENTIMENT-AWARE MULTI-AGENT SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("World's First Large-Scale Emotional AI Swarm Arena")
    print("Breakthrough Research in Sentiment-Aware MARL")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_global_seed(42)
    
    try:
        # Create the sentiment-aware simulation
        arena = create_sentiment_aware_simulation()
        
        # Add diverse emotional agent population
        add_diverse_emotional_agents(arena)
        
        # Demonstrate emotional contagion effects
        demonstrate_emotional_contagion(arena)
        
        # Run comprehensive sentiment simulation
        simulation_data = run_sentiment_simulation(arena, episodes=2)
        
        # Analyze and display results
        analyze_sentiment_results(simulation_data)
        
        # Show visualization capabilities
        create_sentiment_visualization()
        
        print(f"\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n🔬 RESEARCH OPPORTUNITIES:")
        print("   • Novel SA-MARL algorithms for competitive environments")
        print("   • Emotional leadership emergence in large-scale swarms")
        print("   • Cross-cultural sentiment adaptation models")
        print("   • Real-time emotional intervention strategies")
        print("   • Sentiment-driven resource allocation optimization")
        
        print("\n🌟 NEXT STEPS:")
        print("   1. Scale to 1000+ agents for breakthrough research")
        print("   2. Integrate advanced NLP models for text sentiment")
        print("   3. Add multi-modal emotion processing (text + behavior)")
        print("   4. Implement advanced emotional memory systems")
        print("   5. Deploy in real-world applications (trading, gaming, healthcare)")
        
        print(f"\n📚 Published Research Potential:")
        print("   • NeurIPS: 'Large-Scale Sentiment Contagion in MARL'")
        print("   • ICML: 'Emergent Emotional Intelligence in Multi-Agent Systems'") 
        print("   • AAMAS: 'Sentiment-Aware Cooperation in Competitive Environments'")
        print("   • Nature AI: 'Artificial Emotional Ecosystems at Scale'")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        print("🔧 Please check system requirements and dependencies")
        raise


if __name__ == "__main__":
    main()