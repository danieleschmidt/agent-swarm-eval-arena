"""
Adaptive Recovery System with Self-Healing Capabilities.

Advanced recovery system that learns from failures and adapts recovery
strategies using machine learning and pattern recognition.
"""

import time
import pickle
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import json

from ..utils.logging import get_logger
from ..exceptions import RecoveryError

logger = get_logger(__name__)


class FailureType(Enum):
    """Types of system failures."""
    AGENT_CRASH = "agent_crash"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    NETWORK_FAILURE = "network_failure"
    RESOURCE_STARVATION = "resource_starvation"
    SIMULATION_DIVERGENCE = "simulation_divergence"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RESTART_COMPONENT = "restart_component"
    SCALE_RESOURCES = "scale_resources"
    REDISTRIBUTE_LOAD = "redistribute_load"
    ROLLBACK_STATE = "rollback_state"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    QUANTUM_ERROR_CORRECTION = "quantum_error_correction"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class FailureEvent:
    """Represents a system failure event."""
    
    timestamp: float
    failure_type: FailureType
    component: str
    severity: str  # low, medium, high, critical
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class RecoveryAction:
    """Represents a recovery action."""
    
    strategy: RecoveryStrategy
    component: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_time: float = 0.0
    success_probability: float = 0.5
    side_effects: List[str] = field(default_factory=list)


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    
    action: RecoveryAction
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)


class AdaptiveLearningEngine:
    """Machine learning engine for adaptive recovery strategies."""
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 memory_decay: float = 0.95,
                 exploration_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.memory_decay = memory_decay
        self.exploration_rate = exploration_rate
        
        # Strategy effectiveness tracking
        self.strategy_effectiveness = {}
        self.failure_patterns = {}
        self.context_similarity_cache = {}
        
        # Experience replay buffer
        self.experience_buffer = []
        self.max_experience_size = 1000
        
        # Model state
        self.model_state = {
            'weights': {},
            'biases': {},
            'feature_importance': {}
        }
        
    def learn_from_experience(self, 
                            failure: FailureEvent,
                            action: RecoveryAction,
                            result: RecoveryResult) -> None:
        """Learn from recovery experience to improve future decisions."""
        try:
            # Create experience record
            experience = {
                'failure_context': self._extract_failure_features(failure),
                'action_taken': action,
                'success': result.success,
                'execution_time': result.execution_time,
                'timestamp': time.time()
            }
            
            # Add to experience buffer
            self.experience_buffer.append(experience)
            if len(self.experience_buffer) > self.max_experience_size:
                self.experience_buffer = self.experience_buffer[-self.max_experience_size//2:]
            
            # Update strategy effectiveness
            strategy_key = f"{failure.failure_type.value}_{action.strategy.value}"
            
            if strategy_key not in self.strategy_effectiveness:
                self.strategy_effectiveness[strategy_key] = {
                    'success_count': 0,
                    'total_attempts': 0,
                    'avg_execution_time': 0.0,
                    'confidence': 0.5
                }
            
            stats = self.strategy_effectiveness[strategy_key]
            stats['total_attempts'] += 1
            
            if result.success:
                stats['success_count'] += 1
            
            # Update average execution time with exponential moving average
            stats['avg_execution_time'] = (
                0.7 * stats['avg_execution_time'] + 0.3 * result.execution_time
            )
            
            # Update confidence based on success rate
            success_rate = stats['success_count'] / stats['total_attempts']
            stats['confidence'] = success_rate
            
            # Learn failure patterns
            self._update_failure_patterns(failure, result.success)
            
            # Update model weights (simplified neural network approach)
            self._update_model_weights(experience)
            
            logger.info(f"Learned from recovery: {strategy_key}, success: {result.success}")
            
        except Exception as e:
            logger.error(f"Failed to learn from experience: {e}")
    
    def recommend_strategy(self, failure: FailureEvent) -> List[RecoveryAction]:
        """Recommend recovery strategies based on learned experience."""
        try:
            failure_features = self._extract_failure_features(failure)
            
            # Find similar past failures
            similar_experiences = self._find_similar_experiences(failure_features)
            
            # Generate candidate strategies
            candidate_strategies = self._generate_candidate_strategies(failure)
            
            # Score strategies based on learned experience
            scored_strategies = []
            
            for strategy in candidate_strategies:
                score = self._score_strategy(failure, strategy, similar_experiences)
                scored_strategies.append((strategy, score))
            
            # Sort by score (descending)
            scored_strategies.sort(key=lambda x: x[1], reverse=True)
            
            # Return top strategies
            recommended_actions = [strategy for strategy, score in scored_strategies[:3]]
            
            # Add exploration strategy (epsilon-greedy)
            if np.random.random() < self.exploration_rate:
                exploration_strategy = self._generate_exploration_strategy(failure)
                if exploration_strategy:
                    recommended_actions.append(exploration_strategy)
            
            return recommended_actions
            
        except Exception as e:
            logger.error(f"Failed to recommend strategy: {e}")
            return self._generate_fallback_strategies(failure)
    
    def _extract_failure_features(self, failure: FailureEvent) -> Dict[str, float]:
        """Extract numerical features from failure event."""
        features = {
            'failure_type_code': list(FailureType).index(failure.failure_type),
            'severity_code': {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(failure.severity, 2),
            'component_hash': hash(failure.component) % 1000,  # Simplified component encoding
            'time_of_day': (failure.timestamp % 86400) / 86400,  # Normalized time of day
            'description_length': len(failure.description),
            'context_size': len(failure.context)
        }
        
        # Add context-specific features
        if 'cpu_usage' in failure.context:
            features['cpu_usage'] = failure.context['cpu_usage']
        if 'memory_usage' in failure.context:
            features['memory_usage'] = failure.context['memory_usage']
        if 'agent_count' in failure.context:
            features['agent_count'] = failure.context['agent_count']
        
        return features
    
    def _find_similar_experiences(self, 
                                failure_features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Find similar past experiences using feature similarity."""
        similar_experiences = []
        
        for experience in self.experience_buffer:
            similarity = self._calculate_feature_similarity(
                failure_features, 
                experience['failure_context']
            )
            
            if similarity > 0.7:  # Similarity threshold
                experience['similarity'] = similarity
                similar_experiences.append(experience)
        
        # Sort by similarity
        similar_experiences.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_experiences[:10]  # Top 10 similar experiences
    
    def _calculate_feature_similarity(self, 
                                    features1: Dict[str, float],
                                    features2: Dict[str, float]) -> float:
        """Calculate cosine similarity between feature vectors."""
        try:
            # Get common features
            common_features = set(features1.keys()) & set(features2.keys())
            
            if not common_features:
                return 0.0
            
            # Create vectors
            vec1 = np.array([features1[f] for f in common_features])
            vec2 = np.array([features2[f] for f in common_features])
            
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            
            return max(0.0, similarity)
            
        except Exception:
            return 0.0
    
    def _generate_candidate_strategies(self, failure: FailureEvent) -> List[RecoveryAction]:
        """Generate candidate recovery strategies for a failure."""
        strategies = []
        
        # Strategy selection based on failure type
        if failure.failure_type == FailureType.AGENT_CRASH:
            strategies.extend([
                RecoveryAction(
                    strategy=RecoveryStrategy.RESTART_COMPONENT,
                    component=failure.component,
                    parameters={'restart_delay': 1.0},
                    estimated_time=5.0
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.ROLLBACK_STATE,
                    component=failure.component,
                    parameters={'checkpoint_age': 30},
                    estimated_time=10.0
                )
            ])
        
        elif failure.failure_type == FailureType.MEMORY_EXHAUSTION:
            strategies.extend([
                RecoveryAction(
                    strategy=RecoveryStrategy.SCALE_RESOURCES,
                    component=failure.component,
                    parameters={'memory_increase': 0.5},
                    estimated_time=15.0
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                    component=failure.component,
                    parameters={'reduce_agents': 0.2},
                    estimated_time=5.0
                )
            ])
        
        elif failure.failure_type == FailureType.PERFORMANCE_DEGRADATION:
            strategies.extend([
                RecoveryAction(
                    strategy=RecoveryStrategy.REDISTRIBUTE_LOAD,
                    component=failure.component,
                    parameters={'rebalance_factor': 0.3},
                    estimated_time=20.0
                ),
                RecoveryAction(
                    strategy=RecoveryStrategy.SCALE_RESOURCES,
                    component=failure.component,
                    parameters={'cpu_increase': 0.3},
                    estimated_time=10.0
                )
            ])
        
        elif failure.failure_type == FailureType.QUANTUM_DECOHERENCE:
            strategies.append(
                RecoveryAction(
                    strategy=RecoveryStrategy.QUANTUM_ERROR_CORRECTION,
                    component=failure.component,
                    parameters={'correction_cycles': 3},
                    estimated_time=8.0
                )
            )
        
        # Add general fallback strategies
        strategies.extend([
            RecoveryAction(
                strategy=RecoveryStrategy.RESTART_COMPONENT,
                component=failure.component,
                estimated_time=10.0
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                component=failure.component,
                estimated_time=5.0
            )
        ])
        
        return strategies
    
    def _score_strategy(self, 
                       failure: FailureEvent,
                       strategy: RecoveryAction,
                       similar_experiences: List[Dict[str, Any]]) -> float:
        """Score a recovery strategy based on learned experience."""
        try:
            # Base score from strategy effectiveness
            strategy_key = f"{failure.failure_type.value}_{strategy.strategy.value}"
            
            if strategy_key in self.strategy_effectiveness:
                base_score = self.strategy_effectiveness[strategy_key]['confidence']
            else:
                base_score = 0.5  # Default score for unknown strategies
            
            # Adjust score based on similar experiences
            similarity_bonus = 0.0
            for experience in similar_experiences:
                if (experience['action_taken'].strategy == strategy.strategy and
                    experience['success']):
                    similarity_bonus += 0.1 * experience['similarity']
            
            # Time penalty (prefer faster strategies)
            time_penalty = min(0.2, strategy.estimated_time / 60.0)  # Max 20% penalty
            
            # Severity adjustment
            severity_multiplier = {
                'low': 1.0,
                'medium': 1.1,
                'high': 1.2,
                'critical': 1.5
            }.get(failure.severity, 1.0)
            
            final_score = (base_score + similarity_bonus - time_penalty) * severity_multiplier
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Strategy scoring failed: {e}")
            return 0.3  # Conservative fallback score
    
    def _generate_exploration_strategy(self, failure: FailureEvent) -> Optional[RecoveryAction]:
        """Generate an exploration strategy for learning."""
        try:
            # Randomly select an untested strategy combination
            unused_strategies = []
            
            for strategy_type in RecoveryStrategy:
                strategy_key = f"{failure.failure_type.value}_{strategy_type.value}"
                
                if strategy_key not in self.strategy_effectiveness:
                    unused_strategies.append(strategy_type)
            
            if unused_strategies:
                random_strategy = np.random.choice(unused_strategies)
                
                return RecoveryAction(
                    strategy=random_strategy,
                    component=failure.component,
                    parameters={'exploration': True},
                    estimated_time=15.0
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Exploration strategy generation failed: {e}")
            return None
    
    def _generate_fallback_strategies(self, failure: FailureEvent) -> List[RecoveryAction]:
        """Generate safe fallback strategies."""
        return [
            RecoveryAction(
                strategy=RecoveryStrategy.RESTART_COMPONENT,
                component=failure.component,
                estimated_time=10.0
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                component=failure.component,
                estimated_time=5.0
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.EMERGENCY_SHUTDOWN,
                component=failure.component,
                estimated_time=2.0
            )
        ]
    
    def _update_failure_patterns(self, failure: FailureEvent, success: bool) -> None:
        """Update failure pattern recognition."""
        try:
            pattern_key = f"{failure.failure_type.value}_{failure.component}"
            
            if pattern_key not in self.failure_patterns:
                self.failure_patterns[pattern_key] = {
                    'frequency': 0,
                    'last_occurrence': failure.timestamp,
                    'typical_contexts': [],
                    'success_rate': 0.5
                }
            
            pattern = self.failure_patterns[pattern_key]
            pattern['frequency'] += 1
            pattern['last_occurrence'] = failure.timestamp
            
            # Update success rate
            current_rate = pattern['success_rate']
            pattern['success_rate'] = 0.9 * current_rate + 0.1 * (1.0 if success else 0.0)
            
            # Store context for pattern analysis
            if len(pattern['typical_contexts']) < 100:
                pattern['typical_contexts'].append(failure.context)
            
        except Exception as e:
            logger.error(f"Failed to update failure patterns: {e}")
    
    def _update_model_weights(self, experience: Dict[str, Any]) -> None:
        """Update internal model weights (simplified approach)."""
        try:
            # Extract features and target
            features = experience['failure_context']
            target = 1.0 if experience['success'] else 0.0
            
            # Update feature importance
            for feature_name, feature_value in features.items():
                if feature_name not in self.model_state['feature_importance']:
                    self.model_state['feature_importance'][feature_name] = 0.5
                
                # Update importance based on success/failure correlation
                current_importance = self.model_state['feature_importance'][feature_name]
                
                # Simple correlation-based update
                if (feature_value > 0.5 and target > 0.5) or (feature_value <= 0.5 and target <= 0.5):
                    # Positive correlation
                    self.model_state['feature_importance'][feature_name] = min(1.0,
                        current_importance + self.learning_rate * 0.1)
                else:
                    # Negative correlation
                    self.model_state['feature_importance'][feature_name] = max(0.0,
                        current_importance - self.learning_rate * 0.1)
            
        except Exception as e:
            logger.error(f"Model weight update failed: {e}")
    
    def save_model(self, filepath: str) -> None:
        """Save the learned model to disk."""
        try:
            model_data = {
                'strategy_effectiveness': self.strategy_effectiveness,
                'failure_patterns': self.failure_patterns,
                'model_state': self.model_state,
                'experience_buffer': self.experience_buffer[-100:],  # Save recent experiences
                'metadata': {
                    'version': '1.0',
                    'timestamp': time.time(),
                    'total_experiences': len(self.experience_buffer)
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str) -> None:
        """Load a previously saved model."""
        try:
            if not Path(filepath).exists():
                logger.warning(f"Model file {filepath} not found")
                return
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.strategy_effectiveness = model_data.get('strategy_effectiveness', {})
            self.failure_patterns = model_data.get('failure_patterns', {})
            self.model_state = model_data.get('model_state', {'weights': {}, 'biases': {}, 'feature_importance': {}})
            
            # Load recent experiences
            saved_experiences = model_data.get('experience_buffer', [])
            self.experience_buffer.extend(saved_experiences)
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


class AdaptiveRecoverySystem:
    """Main adaptive recovery system with self-healing capabilities."""
    
    def __init__(self, 
                 learning_enabled: bool = True,
                 auto_recovery: bool = True,
                 max_recovery_attempts: int = 3):
        self.learning_enabled = learning_enabled
        self.auto_recovery = auto_recovery
        self.max_recovery_attempts = max_recovery_attempts
        
        # Components
        if learning_enabled:
            self.learning_engine = AdaptiveLearningEngine()
        
        # State tracking
        self.active_failures = {}
        self.recovery_history = []
        self.recovery_executors = {}
        
        # Recovery thread
        self.recovery_thread = None
        self.running = False
        
        # Statistics
        self.stats = {
            'total_failures': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'avg_recovery_time': 0.0,
            'start_time': time.time()
        }
    
    def start(self) -> None:
        """Start the adaptive recovery system."""
        try:
            self.running = True
            
            # Start recovery processing thread
            self.recovery_thread = threading.Thread(
                target=self._process_recoveries,
                daemon=True
            )
            self.recovery_thread.start()
            
            # Load saved model if available
            if self.learning_enabled:
                model_path = "adaptive_recovery_model.pkl"
                self.learning_engine.load_model(model_path)
            
            logger.info("Adaptive recovery system started")
            
        except Exception as e:
            logger.error(f"Failed to start recovery system: {e}")
            raise RecoveryError(f"Recovery system start failed: {e}")
    
    def stop(self) -> None:
        """Stop the adaptive recovery system."""
        try:
            self.running = False
            
            # Save model
            if self.learning_enabled:
                model_path = "adaptive_recovery_model.pkl"
                self.learning_engine.save_model(model_path)
            
            logger.info("Adaptive recovery system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping recovery system: {e}")
    
    def register_failure(self, failure: FailureEvent) -> None:
        """Register a system failure for recovery."""
        try:
            failure_id = f"{failure.component}_{failure.timestamp}"
            self.active_failures[failure_id] = failure
            self.stats['total_failures'] += 1
            
            logger.warning(f"Failure registered: {failure.failure_type.value} in {failure.component}")
            
            # Trigger immediate recovery if auto-recovery is enabled
            if self.auto_recovery:
                self._trigger_recovery(failure_id)
                
        except Exception as e:
            logger.error(f"Failed to register failure: {e}")
    
    def register_recovery_executor(self, 
                                 strategy: RecoveryStrategy,
                                 executor: Callable[[RecoveryAction], RecoveryResult]) -> None:
        """Register a recovery executor for a specific strategy."""
        self.recovery_executors[strategy] = executor
        logger.info(f"Recovery executor registered for {strategy.value}")
    
    def _trigger_recovery(self, failure_id: str) -> None:
        """Trigger recovery for a specific failure."""
        try:
            if failure_id not in self.active_failures:
                return
            
            failure = self.active_failures[failure_id]
            
            # Get recommended strategies
            if self.learning_enabled:
                recommended_actions = self.learning_engine.recommend_strategy(failure)
            else:
                recommended_actions = self._get_default_strategies(failure)
            
            # Attempt recovery with each strategy
            for action in recommended_actions:
                if len(failure.recovery_attempts) >= self.max_recovery_attempts:
                    logger.error(f"Max recovery attempts reached for {failure_id}")
                    break
                
                result = self._execute_recovery_action(action)
                failure.recovery_attempts.append(action.strategy.value)
                
                # Learn from experience
                if self.learning_enabled:
                    self.learning_engine.learn_from_experience(failure, action, result)
                
                # Update statistics
                if result.success:
                    self.stats['successful_recoveries'] += 1
                    failure.resolved = True
                    failure.resolution_time = time.time()
                    
                    # Remove from active failures
                    del self.active_failures[failure_id]
                    break
                else:
                    self.stats['failed_recoveries'] += 1
            
            # Add to history
            self.recovery_history.append(failure)
            
            # Maintain bounded history
            if len(self.recovery_history) > 1000:
                self.recovery_history = self.recovery_history[-500:]
                
        except Exception as e:
            logger.error(f"Recovery trigger failed: {e}")
    
    def _execute_recovery_action(self, action: RecoveryAction) -> RecoveryResult:
        """Execute a recovery action."""
        start_time = time.time()
        
        try:
            # Check if executor is available
            if action.strategy not in self.recovery_executors:
                return RecoveryResult(
                    action=action,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"No executor available for {action.strategy.value}"
                )
            
            # Execute recovery action
            executor = self.recovery_executors[action.strategy]
            result = executor(action)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Update average recovery time
            current_avg = self.stats['avg_recovery_time']
            total_recoveries = self.stats['successful_recoveries'] + self.stats['failed_recoveries']
            
            if total_recoveries > 0:
                self.stats['avg_recovery_time'] = (
                    (current_avg * (total_recoveries - 1) + execution_time) / total_recoveries
                )
            
            logger.info(f"Recovery action executed: {action.strategy.value}, "
                       f"success: {result.success}, time: {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return RecoveryResult(
                action=action,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _get_default_strategies(self, failure: FailureEvent) -> List[RecoveryAction]:
        """Get default recovery strategies when learning is disabled."""
        strategies = []
        
        # Simple rule-based strategy selection
        if failure.failure_type == FailureType.AGENT_CRASH:
            strategies.append(RecoveryAction(
                strategy=RecoveryStrategy.RESTART_COMPONENT,
                component=failure.component
            ))
        elif failure.failure_type == FailureType.MEMORY_EXHAUSTION:
            strategies.append(RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                component=failure.component
            ))
        elif failure.failure_type == FailureType.PERFORMANCE_DEGRADATION:
            strategies.append(RecoveryAction(
                strategy=RecoveryStrategy.REDISTRIBUTE_LOAD,
                component=failure.component
            ))
        
        # Always add restart as fallback
        strategies.append(RecoveryAction(
            strategy=RecoveryStrategy.RESTART_COMPONENT,
            component=failure.component
        ))
        
        return strategies
    
    def _process_recoveries(self) -> None:
        """Process recoveries in background thread."""
        while self.running:
            try:
                # Check for failures that need recovery
                current_time = time.time()
                
                for failure_id, failure in list(self.active_failures.items()):
                    # Skip if already being processed or recently failed
                    if len(failure.recovery_attempts) > 0:
                        last_attempt_time = current_time - 30  # 30 second cooldown
                        if failure.timestamp > last_attempt_time:
                            continue
                    
                    # Trigger recovery for unresolved failures
                    if not failure.resolved:
                        self._trigger_recovery(failure_id)
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Recovery processing error: {e}")
                time.sleep(10.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recovery system statistics."""
        stats = self.stats.copy()
        stats['uptime'] = time.time() - stats['start_time']
        stats['active_failures'] = len(self.active_failures)
        stats['recovery_history_size'] = len(self.recovery_history)
        
        if self.learning_enabled:
            stats['learning_engine'] = {
                'strategy_effectiveness_count': len(self.learning_engine.strategy_effectiveness),
                'failure_patterns_count': len(self.learning_engine.failure_patterns),
                'experience_buffer_size': len(self.learning_engine.experience_buffer)
            }
        
        # Calculate success rate
        total_attempts = stats['successful_recoveries'] + stats['failed_recoveries']
        if total_attempts > 0:
            stats['success_rate'] = stats['successful_recoveries'] / total_attempts
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def get_failure_insights(self) -> Dict[str, Any]:
        """Get insights about failure patterns and recovery effectiveness."""
        insights = {
            'most_common_failures': {},
            'most_effective_strategies': {},
            'failure_trends': {},
            'component_reliability': {}
        }
        
        if not self.recovery_history:
            return insights
        
        # Analyze failure frequency by type
        failure_counts = {}
        for failure in self.recovery_history:
            failure_type = failure.failure_type.value
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
        
        insights['most_common_failures'] = dict(
            sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Analyze strategy effectiveness
        if self.learning_enabled:
            insights['most_effective_strategies'] = {
                k: v['confidence'] 
                for k, v in self.learning_engine.strategy_effectiveness.items()
            }
        
        # Component reliability analysis
        component_stats = {}
        for failure in self.recovery_history:
            component = failure.component
            if component not in component_stats:
                component_stats[component] = {'failures': 0, 'resolved': 0}
            
            component_stats[component]['failures'] += 1
            if failure.resolved:
                component_stats[component]['resolved'] += 1
        
        for component, stats in component_stats.items():
            if stats['failures'] > 0:
                reliability = stats['resolved'] / stats['failures']
                insights['component_reliability'][component] = reliability
        
        return insights