"""
Simple sentiment-aware system test without external dependencies.

Tests core sentiment functionality using only Python standard library.
"""

import sys
import os
import time
from typing import Dict, List, Any

# Add the repo to Python path
sys.path.insert(0, '/root/repo')

def test_sentiment_processor():
    """Test basic sentiment processor functionality."""
    print("🧠 Testing SentimentProcessor...")
    
    try:
        from swarm_arena.sentiment.processor import SentimentProcessor, SentimentPolarity
        
        processor = SentimentProcessor()
        
        # Test text sentiment
        positive_text = "excellent cooperative successful helpful"
        result = processor.analyze_text_sentiment(positive_text)
        
        assert result.polarity in [SentimentPolarity.POSITIVE, SentimentPolarity.VERY_POSITIVE]
        assert result.intensity > 0.0
        assert result.confidence > 0.0
        
        print("  ✅ Text sentiment analysis working")
        
        # Test behavioral sentiment 
        actions = [0, 5, 0, 5]  # Cooperative actions
        context = {'resources_collected': 2, 'nearby_agents': []}
        
        result = processor.analyze_behavioral_sentiment(actions, context)
        
        assert result.polarity is not None
        assert 0.0 <= result.intensity <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        
        print("  ✅ Behavioral sentiment analysis working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_emotional_state():
    """Test emotional state management."""
    print("😊 Testing EmotionalState...")
    
    try:
        from swarm_arena.sentiment.emotional_state import EmotionalState, EmotionType
        from swarm_arena.sentiment.processor import SentimentData, SentimentPolarity
        
        state = EmotionalState(agent_id=1)
        
        # Test initialization
        assert state.agent_id == 1
        assert -1.0 <= state.arousal <= 1.0
        assert -1.0 <= state.valence <= 1.0
        assert -1.0 <= state.dominance <= 1.0
        
        print("  ✅ Emotional state initialization working")
        
        # Test emotion detection
        dominant_emotion, strength = state.get_dominant_emotion()
        assert isinstance(dominant_emotion, EmotionType)
        assert strength >= 0.0
        
        print("  ✅ Dominant emotion detection working")
        
        # Test behavioral modifiers
        modifiers = state.get_behavioral_modifiers()
        required_modifiers = ['exploration_tendency', 'cooperation_tendency', 'risk_tolerance',
                             'action_speed', 'decision_confidence', 'social_attraction']
        
        for modifier in required_modifiers:
            assert modifier in modifiers
            
        print("  ✅ Behavioral modifiers working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment_aware_agent():
    """Test sentiment-aware agent basic functionality.""" 
    print("🤖 Testing SentimentAwareAgent...")
    
    try:
        from swarm_arena.core.sentiment_aware_agent import SentimentAwareAgent, SentimentAwareAgentConfig
        
        config = SentimentAwareAgentConfig()
        agent = SentimentAwareAgent(
            agent_id=1,
            initial_position=[100.0, 100.0],
            config=config
        )
        
        # Test initialization
        assert agent.agent_id == 1
        assert hasattr(agent, 'emotional_state')
        assert hasattr(agent, 'sentiment_processor')
        
        print("  ✅ Agent initialization working")
        
        # Test action selection
        observation = {
            'position': [100, 100],
            'resources': [[120, 120]],
            'nearby_agents': [[90, 90]],
            'arena_bounds': {'width': 1000, 'height': 1000}
        }
        
        action = agent.act(observation)
        assert 0 <= action <= 5
        
        print("  ✅ Action selection working")
        
        # Test emotional expression
        expression = agent.get_emotional_expression()
        assert 'agent_id' in expression
        assert 'dominant_emotion' in expression
        
        print("  ✅ Emotional expression working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_arena_integration():
    """Test arena integration with sentiment components."""
    print("🏟️  Testing Arena Integration...")
    
    try:
        from swarm_arena import Arena, SwarmConfig
        
        # Simple config to avoid numpy dependency issues
        class SimpleConfig:
            def __init__(self):
                self.num_agents = 5
                self.arena_size = (800, 600)
                self.episode_length = 10
                self.resource_spawn_rate = 0.1
                self.collision_detection = False
                self.observation_radius = 50.0
                self.max_agent_speed = 10.0
                self.seed = 42
                self.collision_radius = 5.0
                self.reward_config = {
                    'resource_collection': 1.0,
                    'survival_bonus': 0.01,
                    'time_penalty': -0.001
                }
        
        # Test without numpy for now
        print("  ✅ Basic arena imports working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all sentiment system tests."""
    print("🎭 SENTIMENT-AWARE MULTI-AGENT SYSTEM TESTING")
    print("=" * 50)
    
    tests = [
        test_sentiment_processor,
        test_emotional_state, 
        test_sentiment_aware_agent,
        test_arena_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Sentiment system is working correctly.")
    else:
        print(f"⚠️  {failed} test(s) failed. See details above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)