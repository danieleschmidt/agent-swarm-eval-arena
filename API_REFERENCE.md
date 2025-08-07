# Sentiment-Aware Multi-Agent System API Reference

## 🎯 API Overview

The Sentiment-Aware Multi-Agent Reinforcement Learning (SA-MARL) system provides a comprehensive API for building emotionally intelligent multi-agent simulations. This reference covers all public interfaces, classes, and methods.

## 📚 Core Components

### SentimentProcessor

Primary interface for sentiment analysis operations.

```python
from swarm_arena.sentiment.processor import SentimentProcessor, SentimentData, SentimentPolarity

class SentimentProcessor:
    """Real-time sentiment analysis processor for multi-agent systems."""
    
    def __init__(self, cache_size: int = 1000, enable_caching: bool = True):
        """
        Initialize sentiment processor.
        
        Args:
            cache_size: Maximum number of cached results
            enable_caching: Enable LRU caching for performance
        """
        
    def analyze_text_sentiment(self, text: str) -> SentimentData:
        """
        Analyze sentiment of text input.
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentData: Comprehensive sentiment analysis results
            
        Example:
            >>> processor = SentimentProcessor()
            >>> result = processor.analyze_text_sentiment("excellent teamwork!")
            >>> print(result.polarity.value)  # "POSITIVE"
            >>> print(result.intensity)       # 0.85
        """
        
    def analyze_behavioral_sentiment(self, 
                                   action_sequence: List[int], 
                                   context: Dict[str, Any]) -> SentimentData:
        """
        Infer sentiment from agent behavioral patterns.
        
        Args:
            action_sequence: Recent actions taken by agent
            context: Environmental context (resources, neighbors, etc.)
            
        Returns:
            SentimentData: Behavioral sentiment inference
            
        Example:
            >>> actions = [0, 5, 0, 5]  # Cooperative pattern
            >>> context = {'resources_collected': 3, 'nearby_agents': [1, 2]}
            >>> result = processor.analyze_behavioral_sentiment(actions, context)
            >>> print(result.emotional_dimensions['valence'])  # Positive cooperation
        """
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics."""
```

### EmotionalState

Manages multi-dimensional emotional states for agents.

```python
from swarm_arena.sentiment.emotional_state import EmotionalState, EmotionType, EmotionalContext

class EmotionalState:
    """Multi-dimensional emotional state management for agents."""
    
    def __init__(self, 
                 agent_id: int,
                 initial_arousal: float = 0.0,
                 initial_valence: float = 0.0,
                 initial_dominance: float = 0.0):
        """
        Initialize emotional state.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_arousal: Initial arousal level [-1.0, 1.0]
            initial_valence: Initial valence level [-1.0, 1.0]  
            initial_dominance: Initial dominance level [-1.0, 1.0]
        """
        
    def get_dominant_emotion(self) -> Tuple[EmotionType, float]:
        """
        Get the currently dominant emotion.
        
        Returns:
            Tuple of (EmotionType, strength) where strength is [0.0, 1.0]
            
        Example:
            >>> state = EmotionalState(1, initial_valence=0.8, initial_arousal=0.3)
            >>> emotion, strength = state.get_dominant_emotion()
            >>> print(emotion.value)  # "JOY"
            >>> print(strength)       # 0.76
        """
        
    def get_behavioral_modifiers(self) -> Dict[str, float]:
        """
        Get behavioral modification parameters based on emotional state.
        
        Returns:
            Dictionary of behavioral modifiers:
            - exploration_tendency: [0.0, 1.0]
            - cooperation_tendency: [0.0, 1.0]
            - risk_tolerance: [0.0, 1.0]
            - action_speed: [0.1, 3.0]
            - decision_confidence: [0.0, 1.0]
            - social_attraction: [0.0, 1.0]
            
        Example:
            >>> modifiers = state.get_behavioral_modifiers()
            >>> print(modifiers['cooperation_tendency'])  # 0.73
            >>> print(modifiers['action_speed'])          # 1.15
        """
        
    def update_from_sentiment(self, 
                            sentiment_data: SentimentData,
                            context: EmotionalContext) -> None:
        """
        Update emotional state based on sentiment analysis.
        
        Args:
            sentiment_data: Results from sentiment analysis
            context: Current environmental context
        """
        
    def apply_emotional_decay(self) -> None:
        """Apply natural emotional decay toward neutral state."""
```

### SentimentAwareAgent

Base class for agents with emotional intelligence.

```python
from swarm_arena.core.sentiment_aware_agent import (
    SentimentAwareAgent, SentimentAwareAgentConfig,
    EmotionalCooperativeAgent, EmotionalCompetitiveAgent, EmotionalAdaptiveAgent
)

class SentimentAwareAgent:
    """Base class for sentiment-aware agents with emotional intelligence."""
    
    def __init__(self, 
                 agent_id: int,
                 initial_position: List[float],
                 config: SentimentAwareAgentConfig):
        """
        Initialize sentiment-aware agent.
        
        Args:
            agent_id: Unique identifier
            initial_position: Starting [x, y] coordinates
            config: Agent configuration parameters
        """
        
    def act(self, observation: Dict[str, Any]) -> int:
        """
        Select action based on observation and emotional state.
        
        Args:
            observation: Environment observation including:
                - position: Current [x, y] position
                - resources: List of visible resource positions
                - nearby_agents: List of nearby agent positions
                - arena_bounds: Environment boundaries
                
        Returns:
            int: Action index (0=stay, 1=up, 2=right, 3=down, 4=left, 5=collect)
            
        Example:
            >>> agent = SentimentAwareAgent(1, [100, 100], config)
            >>> obs = {
            ...     'position': [100, 100],
            ...     'resources': [[120, 110]],
            ...     'nearby_agents': [[90, 90]],
            ...     'arena_bounds': {'width': 1000, 'height': 1000}
            ... }
            >>> action = agent.act(obs)
            >>> print(action)  # 2 (move toward resource)
        """
        
    def get_emotional_expression(self) -> Dict[str, Any]:
        """
        Get current emotional expression for visualization.
        
        Returns:
            Dictionary containing:
            - agent_id: Agent identifier
            - dominant_emotion: Current primary emotion
            - emotional_dimensions: Arousal, valence, dominance values
            - behavioral_modifiers: Current behavior modification values
            
        Example:
            >>> expression = agent.get_emotional_expression()
            >>> print(expression['dominant_emotion'])     # "JOY"
            >>> print(expression['emotional_dimensions']) # {'arousal': 0.3, 'valence': 0.8, 'dominance': 0.1}
        """
        
    def process_emotional_influence(self, influences: List[Dict[str, Any]]) -> None:
        """Process emotional influences from nearby agents."""
```

### SentimentContagion

Manages emotional spread between agents.

```python
from swarm_arena.sentiment.contagion import SentimentContagion, ContagionParameters

class SentimentContagion:
    """Emotional contagion simulation for multi-agent systems."""
    
    def __init__(self, parameters: ContagionParameters):
        """
        Initialize contagion system.
        
        Args:
            parameters: Contagion behavior parameters
        """
        
    def process_emotional_contagion(self,
                                  agent_emotions: Dict[int, EmotionalState],
                                  agent_positions: Dict[int, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process emotional influence propagation between agents.
        
        Args:
            agent_emotions: Current emotional states by agent ID
            agent_positions: Agent positions for proximity calculation
            
        Returns:
            List of emotional influence events
            
        Example:
            >>> contagion = SentimentContagion(ContagionParameters())
            >>> emotions = {1: emotional_state_1, 2: emotional_state_2}
            >>> positions = {1: np.array([100, 100]), 2: np.array([120, 110])}
            >>> influences = contagion.process_emotional_contagion(emotions, positions)
            >>> print(len(influences))  # Number of influence events
        """
        
    def get_contagion_statistics(self) -> Dict[str, Any]:
        """Get statistics about emotional contagion processes."""
```

### Arena Integration

Main simulation environment with sentiment support.

```python
from swarm_arena import Arena, SwarmConfig

class Arena:
    """Multi-agent simulation arena with sentiment awareness."""
    
    def __init__(self, 
                 config: SwarmConfig,
                 sentiment_enabled: bool = True,
                 performance_optimizer: Optional[PerformanceOptimizer] = None):
        """
        Initialize simulation arena.
        
        Args:
            config: Arena configuration
            sentiment_enabled: Enable sentiment processing
            performance_optimizer: Optional performance optimization
        """
        
    def add_sentiment_aware_agents(self, agents: List[SentimentAwareAgent]) -> None:
        """Add sentiment-aware agents to simulation."""
        
    def step(self) -> Dict[str, Any]:
        """
        Execute one simulation step.
        
        Returns:
            Dictionary containing:
            - step_count: Current step number
            - agent_states: Current agent positions and states
            - sentiment_metrics: Aggregated sentiment data
            - performance_metrics: Processing performance data
        """
        
    def get_sentiment_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis of current simulation state.
        
        Returns:
            Dictionary containing:
            - sentiment_distribution: Count by sentiment type
            - emotional_contagion_events: Recent influence events
            - cooperation_metrics: Agent cooperation statistics
            - behavioral_patterns: Detected behavioral trends
        """
```

## 🌍 Global and Cultural Features

### Internationalization Support

```python
from swarm_arena.utils.i18n import (
    GlobalSentimentManager, LocalizationConfig, 
    Language, CulturalContext
)

class GlobalSentimentManager:
    """Multi-language and culturally-aware sentiment analysis."""
    
    def __init__(self, config: LocalizationConfig):
        """
        Initialize global sentiment manager.
        
        Args:
            config: Localization configuration
            
        Example:
            >>> config = LocalizationConfig(
            ...     primary_language=Language.SPANISH,
            ...     cultural_context=CulturalContext.LATIN_AMERICAN,
            ...     enable_cultural_adaptation=True
            ... )
            >>> manager = GlobalSentimentManager(config)
        """
        
    def analyze_sentiment(self, 
                         text: str, 
                         language: Optional[Language] = None) -> Dict[str, Any]:
        """
        Analyze sentiment with cultural and linguistic adaptation.
        
        Args:
            text: Input text in specified language
            language: Target language (defaults to primary_language)
            
        Returns:
            Culturally-adapted sentiment analysis results
        """
        
    def adapt_emotional_behavior(self, 
                               base_emotions: Dict[str, float]) -> Dict[str, float]:
        """Adapt emotional behavior based on cultural context."""
```

## ⚡ Performance and Scaling

### Performance Optimization

```python
from swarm_arena.utils.performance_optimizer import (
    PerformanceOptimizer, PerformanceConfig
)

class PerformanceOptimizer:
    """Performance optimization for large-scale simulations."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """Initialize performance optimizer."""
        
    def get_cached_sentiment(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached sentiment result."""
        
    def cache_sentiment(self, cache_key: str, result: Any) -> None:
        """Store sentiment result in cache."""
        
    def get_nearby_agents(self, 
                         position: Tuple[float, float], 
                         radius: float) -> List[int]:
        """Get nearby agents using spatial optimization."""
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
```

### Distributed Computing

```python
from swarm_arena.utils.distributed_computing import (
    DistributedSentimentManager, DistributedConfig
)

class DistributedSentimentManager:
    """Distributed sentiment processing across multiple workers."""
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        """Initialize distributed computing manager."""
        
    def initialize(self) -> bool:
        """Initialize Ray cluster and workers."""
        
    def process_text_sentiments_distributed(self, 
                                          texts: List[str]) -> List[Dict[str, Any]]:
        """Process text sentiments using distributed workers."""
        
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about Ray cluster status."""
        
    def get_worker_stats(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get statistics from all distributed workers."""
```

## 📊 Monitoring and Telemetry

### Sentiment Telemetry

```python
from swarm_arena.monitoring.sentiment_telemetry import (
    SentimentTelemetryCollector, SentimentMetrics
)

class SentimentTelemetryCollector:
    """Collect and export sentiment-related metrics."""
    
    def collect_agent_sentiment_metrics(self, 
                                       agents: List[SentimentAwareAgent]) -> SentimentMetrics:
        """Collect current sentiment metrics from all agents."""
        
    def export_metrics(self, metrics: SentimentMetrics) -> None:
        """Export metrics to configured monitoring systems."""
        
    def get_aggregated_analytics(self) -> Dict[str, Any]:
        """Get aggregated sentiment analytics over time."""
```

## 🧪 Testing and Validation

### Validation Functions

```python
# Run comprehensive system validation
def validate_system() -> bool:
    """
    Run complete system validation.
    
    Returns:
        bool: True if all validation checks pass
        
    Example:
        >>> from validate_system import main as validate
        >>> success = validate()
        >>> print("System validated!" if success else "Validation failed!")
    """

# Test individual components
def test_sentiment_processor() -> bool:
    """Test sentiment processor functionality."""
    
def test_emotional_state() -> bool:
    """Test emotional state management."""
    
def test_sentiment_aware_agent() -> bool:
    """Test sentiment-aware agent behavior."""
```

## 🔧 Configuration Options

### Core Configuration Classes

```python
from swarm_arena.core.config import SwarmConfig

@dataclass
class SwarmConfig:
    """Main configuration for swarm arena."""
    num_agents: int = 100
    arena_size: Tuple[int, int] = (800, 600)
    episode_length: int = 1000
    resource_spawn_rate: float = 0.1
    max_agent_speed: float = 5.0
    collision_detection: bool = True
    observation_radius: float = 50.0
    
@dataclass
class SentimentAwareAgentConfig:
    """Configuration for sentiment-aware agents."""
    sentiment_update_frequency: float = 1.0
    emotional_decay_rate: float = 0.01
    contagion_susceptibility: float = 0.5
    behavioral_adaptation_rate: float = 0.1
    memory_capacity: int = 100
    
@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    enable_sentiment_caching: bool = True
    sentiment_cache_size: int = 2000
    cache_ttl_seconds: float = 30.0
    enable_batch_processing: bool = True
    batch_size: int = 50
    enable_multithreading: bool = True
    max_worker_threads: int = 4
```

## 📋 Data Structures

### SentimentData

```python
@dataclass
class SentimentData:
    """Comprehensive sentiment analysis results."""
    polarity: SentimentPolarity          # POSITIVE/NEGATIVE/NEUTRAL/etc.
    intensity: float                     # [0.0, 1.0]
    confidence: float                    # [0.0, 1.0]
    emotional_dimensions: Dict[str, float]  # arousal, valence, dominance
    processing_time: float               # milliseconds
    metadata: Dict[str, Any]            # additional analysis data
```

### Enumerations

```python
class SentimentPolarity(Enum):
    """Sentiment polarity classifications."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

class EmotionType(Enum):
    """Primary emotion types based on Plutchik's wheel."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"

class Language(Enum):
    """Supported languages for sentiment analysis."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    # ... additional languages

class CulturalContext(Enum):
    """Cultural contexts for emotional expression."""
    WESTERN_INDIVIDUALISTIC = "western_ind"
    WESTERN_COLLECTIVISTIC = "western_col"
    EAST_ASIAN = "east_asian"
    LATIN_AMERICAN = "latin_am"
    # ... additional contexts
```

## 🔄 Event System

### Event Types

```python
# Sentiment events
class SentimentEvent:
    agent_id: int
    old_sentiment: SentimentData
    new_sentiment: SentimentData
    timestamp: float
    trigger: str  # "text_analysis", "behavioral_inference", "contagion"

# Emotional contagion events
class ContagionEvent:
    source_agent_id: int
    target_agent_id: int
    influence_strength: float
    distance: float
    emotional_change: Dict[str, float]
    timestamp: float

# Behavioral events
class BehaviorEvent:
    agent_id: int
    action: int
    emotional_state: Dict[str, float]
    behavioral_modifiers: Dict[str, float]
    context: Dict[str, Any]
    timestamp: float
```

## 📖 Usage Examples

### Basic Sentiment Analysis

```python
from swarm_arena.sentiment.processor import SentimentProcessor

# Initialize processor
processor = SentimentProcessor()

# Analyze text sentiment
result = processor.analyze_text_sentiment("Excellent teamwork and collaboration!")
print(f"Sentiment: {result.polarity.value}")
print(f"Intensity: {result.intensity:.2f}")
print(f"Confidence: {result.confidence:.2f}")

# Analyze behavioral sentiment
actions = [0, 5, 0, 5, 0]  # Cooperative pattern
context = {
    'resources_collected': 3,
    'nearby_agents': [2, 3, 4],
    'recent_interactions': ['cooperative', 'helpful']
}
behavioral_result = processor.analyze_behavioral_sentiment(actions, context)
print(f"Behavioral sentiment: {behavioral_result.polarity.value}")
```

### Creating Sentiment-Aware Agents

```python
from swarm_arena.core.sentiment_aware_agent import (
    SentimentAwareAgent, SentimentAwareAgentConfig,
    EmotionalCooperativeAgent
)

# Configure agent
config = SentimentAwareAgentConfig(
    sentiment_update_frequency=2.0,  # Update every 2 seconds
    emotional_decay_rate=0.02,       # Faster emotional decay
    contagion_susceptibility=0.7,    # Highly susceptible to emotional contagion
    behavioral_adaptation_rate=0.15  # Rapid behavioral adaptation
)

# Create cooperative agent with emotional intelligence
agent = EmotionalCooperativeAgent(
    agent_id=1,
    initial_position=[100.0, 100.0],
    config=config
)

# Create environment observation
observation = {
    'position': [100, 100],
    'resources': [[150, 120], [80, 90]],
    'nearby_agents': [[110, 105], [95, 95]],
    'arena_bounds': {'width': 1000, 'height': 1000}
}

# Agent selects action based on observation and emotional state
action = agent.act(observation)
print(f"Selected action: {action}")

# Get emotional expression
expression = agent.get_emotional_expression()
print(f"Dominant emotion: {expression['dominant_emotion']}")
print(f"Cooperation tendency: {expression['behavioral_modifiers']['cooperation_tendency']:.2f}")
```

### Running Sentiment-Aware Simulation

```python
from swarm_arena import Arena, SwarmConfig
from swarm_arena.core.sentiment_aware_agent import EmotionalCooperativeAgent, EmotionalCompetitiveAgent
from swarm_arena.utils.performance_optimizer import PerformanceOptimizer

# Configure simulation
config = SwarmConfig(
    num_agents=200,
    arena_size=(1500, 1200),
    episode_length=2000,
    resource_spawn_rate=0.08,
    max_agent_speed=12.0,
    observation_radius=75.0
)

# Initialize performance optimization
optimizer = PerformanceOptimizer()
optimizer.initialize_spatial_grid(config.arena_size[0], config.arena_size[1])

# Create arena with sentiment support
arena = Arena(config, sentiment_enabled=True, performance_optimizer=optimizer)

# Create diverse agent population
agents = []
agent_config = SentimentAwareAgentConfig()

# 60% cooperative agents
for i in range(120):
    agent = EmotionalCooperativeAgent(i, [random.randint(0, 1500), random.randint(0, 1200)], agent_config)
    agents.append(agent)

# 40% competitive agents  
for i in range(120, 200):
    agent = EmotionalCompetitiveAgent(i, [random.randint(0, 1500), random.randint(0, 1200)], agent_config)
    agents.append(agent)

# Add agents to arena
arena.add_sentiment_aware_agents(agents)

# Run simulation
for episode in range(100):
    step_data = arena.step()
    
    # Get sentiment analytics every 10 steps
    if arena.current_step % 10 == 0:
        analytics = arena.get_sentiment_analytics()
        print(f"Episode {episode}, Step {arena.current_step}")
        print(f"  Sentiment distribution: {analytics['sentiment_distribution']}")
        print(f"  Cooperation rate: {analytics['cooperation_metrics']['cooperation_rate']:.2f}")
        print(f"  Contagion events: {len(analytics['emotional_contagion_events'])}")

# Get final performance statistics
perf_stats = optimizer.get_performance_stats()
print(f"\nPerformance Summary:")
print(f"  Cache hit rate: {perf_stats['cache_hit_rate']:.2%}")
print(f"  Average processing time: {perf_stats['avg_processing_time_ms']:.2f}ms")
print(f"  Spatial queries: {perf_stats['spatial_queries']}")
```

### Distributed Processing Setup

```python
from swarm_arena.utils.distributed_computing import DistributedSentimentManager, DistributedConfig

# Configure distributed processing
distributed_config = DistributedConfig(
    enable_distributed=True,
    ray_address="ray://ray-head:10001",  # Connect to Ray cluster
    num_sentiment_workers=8,
    num_contagion_workers=4,
    worker_cpu_allocation=2.0,
    worker_memory_allocation="4GB"
)

# Initialize distributed manager
dist_manager = DistributedSentimentManager(distributed_config)
success = dist_manager.initialize()

if success:
    print("Distributed computing initialized successfully")
    
    # Process large batch of text sentiments
    texts = ["sentiment text"] * 1000  # Large batch
    results = dist_manager.process_text_sentiments_distributed(texts)
    
    # Get worker statistics
    worker_stats = dist_manager.get_worker_stats()
    print(f"Active sentiment workers: {len(worker_stats['sentiment_workers'])}")
    
    # Get cluster information
    cluster_info = dist_manager.get_cluster_info()
    print(f"Cluster nodes: {cluster_info['num_nodes']}")
    print(f"Available resources: {cluster_info['available_resources']}")
    
else:
    print("Failed to initialize distributed computing")
```

---

## 📞 Support and Contributing

- **GitHub Issues:** [Report bugs and request features](https://github.com/danieleschmidt/sentiment-analyzer-pro/issues)
- **Documentation:** [Full documentation](https://github.com/danieleschmidt/sentiment-analyzer-pro/docs)
- **Contributing Guide:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **License:** [LICENSE](LICENSE)

🎭 **Sentiment-Aware Multi-Agent System**  
Version 1.0.0 | API Reference | Research Platform