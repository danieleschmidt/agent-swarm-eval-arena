"""
Neural Swarm Intelligence: Next-generation collective intelligence algorithms.

This module implements breakthrough neural architectures for swarm intelligence,
including transformer-based collective decision making and neural emergence detection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import math
from abc import ABC, abstractmethod

@dataclass
class SwarmIntelligenceConfig:
    """Configuration for neural swarm intelligence."""
    embedding_dim: int = 128
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    max_agents: int = 1000
    attention_mechanism: str = "scaled_dot_product"  # or "sparse", "local"
    
@dataclass
class CollectiveDecision:
    """Result of collective decision making."""
    decision_vector: np.ndarray
    confidence: float
    consensus_level: float
    participating_agents: List[int]
    decision_path: List[Dict[str, Any]]

class MultiAgentTransformer(nn.Module):
    """Transformer architecture for multi-agent collective intelligence."""
    
    def __init__(self, config: SwarmIntelligenceConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        # Positional encoding for agent positions
        self.positional_encoding = PositionalEncoding(config.embedding_dim, config.max_agents)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiAgentAttentionLayer(config) for _ in range(config.num_layers)
        ])
        
        # Output projection for collective decision
        self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.decision_head = nn.Linear(config.embedding_dim, 1)
        
        # Consensus measurement
        self.consensus_head = nn.Linear(config.embedding_dim, 1)
        
    def forward(self, 
                agent_embeddings: torch.Tensor,
                agent_positions: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for collective decision making.
        
        Args:
            agent_embeddings: (batch_size, num_agents, embedding_dim)
            agent_positions: (batch_size, num_agents, 2) - spatial positions
            attention_mask: (batch_size, num_agents) - mask for valid agents
            
        Returns:
            Dictionary containing decision outputs and attention weights
        """
        batch_size, num_agents, _ = agent_embeddings.shape
        
        # Project input embeddings
        x = self.input_projection(agent_embeddings)
        
        # Add positional encoding if positions provided
        if agent_positions is not None:
            pos_encoding = self.positional_encoding(agent_positions)
            x = x + pos_encoding
        
        # Store attention weights for analysis
        attention_weights = []
        
        # Pass through attention layers
        for layer in self.attention_layers:
            x, attn_weights = layer(x, attention_mask)
            attention_weights.append(attn_weights)
        
        # Generate collective decision
        collective_representation = torch.mean(x, dim=1)  # Global aggregation
        decision_logits = self.decision_head(collective_representation)
        consensus_logits = self.consensus_head(collective_representation)
        
        # Calculate per-agent contributions
        agent_contributions = self.output_projection(x)
        
        return {
            'collective_decision': torch.sigmoid(decision_logits),
            'consensus_score': torch.sigmoid(consensus_logits),
            'agent_contributions': agent_contributions,
            'attention_weights': attention_weights,
            'final_representation': x
        }

class PositionalEncoding(nn.Module):
    """Positional encoding for agent spatial positions."""
    
    def __init__(self, embedding_dim: int, max_agents: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.position_encoder = nn.Linear(2, embedding_dim)
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode spatial positions into embeddings.
        
        Args:
            positions: (batch_size, num_agents, 2)
            
        Returns:
            Positional embeddings: (batch_size, num_agents, embedding_dim)
        """
        return self.position_encoder(positions)

class MultiAgentAttentionLayer(nn.Module):
    """Multi-head attention layer optimized for agent interactions."""
    
    def __init__(self, config: SwarmIntelligenceConfig):
        super().__init__()
        self.config = config
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, 
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention layer.
        
        Args:
            x: (batch_size, num_agents, embedding_dim)
            attention_mask: (batch_size, num_agents)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention
        attn_output, attn_weights = self.multihead_attn(
            x, x, x, key_padding_mask=attention_mask
        )
        
        # Residual connection and normalization
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x, attn_weights

class EmergenceDetectionNetwork(nn.Module):
    """Neural network for detecting emergent behaviors in real-time."""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        # Temporal convolutional layers for pattern detection
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU()
        )
        
        # Attention mechanism for important time steps
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Classification heads for different emergence types
        self.emergence_classifiers = nn.ModuleDict({
            'flocking': nn.Linear(hidden_dim, 2),
            'clustering': nn.Linear(hidden_dim, 2),
            'leadership': nn.Linear(hidden_dim, 2),
            'synchronization': nn.Linear(hidden_dim, 2),
            'phase_transition': nn.Linear(hidden_dim, 2)
        })
        
        # Confidence estimation
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, temporal_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect emergent behaviors from temporal features.
        
        Args:
            temporal_features: (batch_size, sequence_length, feature_dim)
            
        Returns:
            Dictionary of emergence probabilities and confidence scores
        """
        batch_size, seq_len, feature_dim = temporal_features.shape
        
        # Temporal convolution (transpose for conv1d)
        x = temporal_features.transpose(1, 2)  # (batch, feature_dim, seq_len)
        conv_features = self.temporal_conv(x)
        conv_features = conv_features.transpose(1, 2)  # Back to (batch, seq_len, hidden_dim)
        
        # Temporal attention
        attended_features, _ = self.temporal_attention(
            conv_features, conv_features, conv_features
        )
        
        # Global pooling
        global_features = torch.mean(attended_features, dim=1)  # (batch, hidden_dim)
        
        # Emergence classification
        emergence_probs = {}
        for emergence_type, classifier in self.emergence_classifiers.items():
            logits = classifier(global_features)
            emergence_probs[emergence_type] = torch.softmax(logits, dim=-1)[:, 1]  # Probability of emergence
        
        # Confidence estimation
        confidence = torch.sigmoid(self.confidence_head(global_features)).squeeze(-1)
        
        return {
            'emergence_probabilities': emergence_probs,
            'confidence': confidence,
            'global_features': global_features
        }

class NeuralSwarmIntelligence:
    """Main class for neural swarm intelligence algorithms."""
    
    def __init__(self, config: SwarmIntelligenceConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.collective_decision_network = MultiAgentTransformer(config)
        self.emergence_detector = EmergenceDetectionNetwork()
        
        # Move to device
        self.collective_decision_network.to(self.device)
        self.emergence_detector.to(self.device)
        
        # Optimizers
        self.optimizer = torch.optim.AdamW(
            list(self.collective_decision_network.parameters()) +
            list(self.emergence_detector.parameters()),
            lr=1e-4, weight_decay=1e-5
        )
        
    def collective_decision_making(self,
                                 agent_states: Dict[int, np.ndarray],
                                 agent_positions: Dict[int, np.ndarray],
                                 decision_context: Optional[np.ndarray] = None) -> CollectiveDecision:
        """
        Perform collective decision making using neural transformer.
        
        Args:
            agent_states: Dict mapping agent_id to state vector
            agent_positions: Dict mapping agent_id to position
            decision_context: Optional global context for decision
            
        Returns:
            CollectiveDecision with neural-enhanced decision making
        """
        self.collective_decision_network.eval()
        
        with torch.no_grad():
            # Prepare input tensors
            agent_ids = list(agent_states.keys())
            num_agents = len(agent_ids)
            
            if num_agents == 0:
                return CollectiveDecision(
                    decision_vector=np.array([0.5]),
                    confidence=0.0,
                    consensus_level=0.0,
                    participating_agents=[],
                    decision_path=[]
                )
            
            # Stack agent states and positions
            states = np.stack([agent_states[aid] for aid in agent_ids])
            positions = np.stack([agent_positions[aid] for aid in agent_ids])
            
            # Convert to tensors
            states_tensor = torch.FloatTensor(states).unsqueeze(0).to(self.device)
            positions_tensor = torch.FloatTensor(positions).unsqueeze(0).to(self.device)
            
            # Ensure correct embedding dimension
            if states_tensor.shape[-1] != self.config.embedding_dim:
                # Project to correct dimension
                states_tensor = F.linear(
                    states_tensor,
                    torch.randn(self.config.embedding_dim, states_tensor.shape[-1]).to(self.device)
                )
            
            # Forward pass
            outputs = self.collective_decision_network(
                agent_embeddings=states_tensor,
                agent_positions=positions_tensor
            )
            
            # Extract results
            decision_vector = outputs['collective_decision'].cpu().numpy().flatten()
            consensus_score = outputs['consensus_score'].cpu().numpy().flatten()[0]
            
            # Calculate participation based on attention weights
            attention_weights = outputs['attention_weights'][-1]  # Use last layer
            avg_attention = torch.mean(attention_weights, dim=1)  # Average over heads
            participation_scores = torch.mean(avg_attention, dim=1).cpu().numpy()  # Average over targets
            
            # Determine participating agents (above threshold)
            participation_threshold = 0.1
            participating_agents = [
                agent_ids[i] for i, score in enumerate(participation_scores)
                if score > participation_threshold
            ]
            
            # Build decision path
            decision_path = [{
                'layer': i,
                'attention_pattern': attn.cpu().numpy().tolist()
            } for i, attn in enumerate(attention_weights)]
            
            return CollectiveDecision(
                decision_vector=decision_vector,
                confidence=float(np.mean(decision_vector)),
                consensus_level=float(consensus_score),
                participating_agents=participating_agents,
                decision_path=decision_path
            )
    
    def detect_emergent_intelligence(self,
                                   agent_trajectories: Dict[int, np.ndarray],
                                   time_window: int = 50) -> Dict[str, Any]:
        """
        Detect emergent intelligence patterns using neural networks.
        
        Args:
            agent_trajectories: Historical agent trajectories
            time_window: Time window for analysis
            
        Returns:
            Dictionary of detected emergent intelligence patterns
        """
        self.emergence_detector.eval()
        
        with torch.no_grad():
            # Extract temporal features
            temporal_features = self._extract_temporal_features(
                agent_trajectories, time_window
            )
            
            if temporal_features is None:
                return {
                    'emergent_patterns': {},
                    'confidence': 0.0,
                    'intelligence_metrics': {}
                }
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(temporal_features).unsqueeze(0).to(self.device)
            
            # Detect emergence
            outputs = self.emergence_detector(features_tensor)
            
            # Extract results
            emergence_probs = {
                pattern_type: float(prob.cpu().numpy())
                for pattern_type, prob in outputs['emergence_probabilities'].items()
            }
            
            confidence = float(outputs['confidence'].cpu().numpy())
            
            # Calculate intelligence metrics
            intelligence_metrics = self._calculate_intelligence_metrics(
                emergence_probs, confidence
            )
            
            return {
                'emergent_patterns': emergence_probs,
                'confidence': confidence,
                'intelligence_metrics': intelligence_metrics,
                'neural_features': outputs['global_features'].cpu().numpy()
            }
    
    def _extract_temporal_features(self,
                                 agent_trajectories: Dict[int, np.ndarray],
                                 time_window: int) -> Optional[np.ndarray]:
        """Extract temporal features from agent trajectories."""
        if not agent_trajectories:
            return None
            
        agent_ids = list(agent_trajectories.keys())
        
        # Find common time length
        min_length = min(len(traj) for traj in agent_trajectories.values())
        
        if min_length < time_window:
            return None
            
        # Extract features for each time step
        temporal_features = []
        
        for t in range(min_length - time_window, min_length):
            step_features = []
            
            # Spatial features
            positions = [agent_trajectories[aid][t] for aid in agent_ids]
            positions_array = np.array(positions)
            
            # Calculate spatial statistics
            centroid = np.mean(positions_array, axis=0)
            spread = np.std(positions_array, axis=0)
            
            # Pairwise distances
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            
            # Aggregate distance statistics
            if distances:
                dist_stats = [
                    np.mean(distances),
                    np.std(distances),
                    np.min(distances),
                    np.max(distances)
                ]
            else:
                dist_stats = [0.0, 0.0, 0.0, 0.0]
            
            # Velocity features (if trajectory has enough points)
            if t > 0:
                velocities = []
                for aid in agent_ids:
                    vel = agent_trajectories[aid][t] - agent_trajectories[aid][t-1]
                    velocities.append(vel)
                
                velocities_array = np.array(velocities)
                velocity_stats = [
                    np.mean(velocities_array),
                    np.std(velocities_array),
                    np.linalg.norm(np.mean(velocities_array, axis=0))  # Collective velocity magnitude
                ]
            else:
                velocity_stats = [0.0, 0.0, 0.0]
            
            # Combine all features
            step_features.extend(centroid.tolist())
            step_features.extend(spread.tolist())
            step_features.extend(dist_stats)
            step_features.extend(velocity_stats)
            
            # Pad or truncate to fixed size
            feature_size = 128  # Fixed feature dimension
            if len(step_features) < feature_size:
                step_features.extend([0.0] * (feature_size - len(step_features)))
            else:
                step_features = step_features[:feature_size]
            
            temporal_features.append(step_features)
        
        return np.array(temporal_features)
    
    def _calculate_intelligence_metrics(self,
                                      emergence_probs: Dict[str, float],
                                      confidence: float) -> Dict[str, float]:
        """Calculate intelligence metrics from emergence patterns."""
        metrics = {}
        
        # Collective intelligence index
        metrics['collective_intelligence_index'] = (
            0.3 * emergence_probs.get('synchronization', 0.0) +
            0.25 * emergence_probs.get('leadership', 0.0) +
            0.2 * emergence_probs.get('flocking', 0.0) +
            0.15 * emergence_probs.get('clustering', 0.0) +
            0.1 * emergence_probs.get('phase_transition', 0.0)
        ) * confidence
        
        # Adaptation capability
        metrics['adaptation_capability'] = emergence_probs.get('phase_transition', 0.0) * confidence
        
        # Coordination efficiency
        metrics['coordination_efficiency'] = (
            emergence_probs.get('synchronization', 0.0) + 
            emergence_probs.get('flocking', 0.0)
        ) / 2.0 * confidence
        
        # Leadership emergence
        metrics['leadership_emergence'] = emergence_probs.get('leadership', 0.0) * confidence
        
        # Overall swarm intelligence
        metrics['swarm_intelligence_score'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def train_on_trajectories(self,
                            training_trajectories: List[Dict[int, np.ndarray]],
                            labels: List[Dict[str, float]],
                            epochs: int = 100) -> Dict[str, List[float]]:
        """
        Train the neural networks on trajectory data.
        
        Args:
            training_trajectories: List of trajectory dictionaries
            labels: List of ground truth labels for emergence patterns
            epochs: Number of training epochs
            
        Returns:
            Training history with losses and metrics
        """
        self.collective_decision_network.train()
        self.emergence_detector.train()
        
        history = {'loss': [], 'emergence_accuracy': []}
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for trajectories, label in zip(training_trajectories, labels):
                # Prepare data
                temporal_features = self._extract_temporal_features(trajectories, 50)
                
                if temporal_features is None:
                    continue
                
                features_tensor = torch.FloatTensor(temporal_features).unsqueeze(0).to(self.device)
                
                # Forward pass through emergence detector
                outputs = self.emergence_detector(features_tensor)
                
                # Calculate loss
                loss = 0.0
                for pattern_type, prob in outputs['emergence_probabilities'].items():
                    if pattern_type in label:
                        target = torch.FloatTensor([label[pattern_type]]).to(self.device)
                        loss += F.binary_cross_entropy(prob.unsqueeze(0), target)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                for pattern_type, prob in outputs['emergence_probabilities'].items():
                    if pattern_type in label:
                        predicted = (prob > 0.5).float()
                        actual = label[pattern_type]
                        if (predicted.item() > 0.5) == (actual > 0.5):
                            correct_predictions += 1
                        total_predictions += 1
            
            # Record history
            avg_loss = total_loss / len(training_trajectories)
            accuracy = correct_predictions / max(total_predictions, 1)
            
            history['loss'].append(avg_loss)
            history['emergence_accuracy'].append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        return history