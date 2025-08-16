"""
Neuromorphic Processing for Swarm Intelligence

This module implements neuromorphic computing paradigms for ultra-efficient
real-time processing of swarm behaviors and decision making.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import threading
import time
from enum import Enum
import json

class SpikePattern(Enum):
    """Types of spike patterns in neuromorphic processing."""
    REGULAR = "regular"
    BURST = "burst"
    IRREGULAR = "irregular"
    SYNCHRONIZED = "synchronized"
    CHAOTIC = "chaotic"

@dataclass
class Spike:
    """Individual spike event."""
    neuron_id: int
    timestamp: float
    amplitude: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SpikeTrain:
    """Sequence of spikes from a neuron."""
    neuron_id: int
    spikes: List[Spike]
    start_time: float
    end_time: float
    
    def spike_rate(self) -> float:
        """Calculate average spike rate."""
        duration = self.end_time - self.start_time
        return len(self.spikes) / duration if duration > 0 else 0.0
    
    def inter_spike_intervals(self) -> List[float]:
        """Calculate inter-spike intervals."""
        if len(self.spikes) < 2:
            return []
        
        intervals = []
        for i in range(1, len(self.spikes)):
            interval = self.spikes[i].timestamp - self.spikes[i-1].timestamp
            intervals.append(interval)
        
        return intervals

class SpikingNeuron:
    """Leaky Integrate-and-Fire neuron model."""
    
    def __init__(self, 
                 neuron_id: int,
                 threshold: float = 1.0,
                 reset_potential: float = 0.0,
                 membrane_resistance: float = 1.0,
                 membrane_capacitance: float = 1.0,
                 refractory_period: float = 0.001):
        
        self.neuron_id = neuron_id
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.membrane_resistance = membrane_resistance
        self.membrane_capacitance = membrane_capacitance
        self.refractory_period = refractory_period
        
        # State variables
        self.membrane_potential = reset_potential
        self.last_spike_time = -float('inf')
        self.input_current = 0.0
        
        # Spike history
        self.spike_history = []
        
    def update(self, dt: float, input_current: float) -> Optional[Spike]:
        """Update neuron state and return spike if generated."""
        current_time = time.time()
        
        # Check refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            return None
        
        # Update membrane potential using Euler integration
        tau = self.membrane_resistance * self.membrane_capacitance
        
        dv_dt = (-self.membrane_potential + self.membrane_resistance * input_current) / tau
        self.membrane_potential += dv_dt * dt
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            # Generate spike
            spike = Spike(
                neuron_id=self.neuron_id,
                timestamp=current_time,
                amplitude=self.membrane_potential
            )
            
            # Reset membrane potential
            self.membrane_potential = self.reset_potential
            self.last_spike_time = current_time
            
            # Record spike
            self.spike_history.append(spike)
            
            return spike
        
        return None
    
    def inject_current(self, current: float):
        """Inject current into the neuron."""
        self.input_current = current

class SynapticConnection:
    """Synaptic connection between neurons."""
    
    def __init__(self,
                 pre_neuron_id: int,
                 post_neuron_id: int,
                 weight: float = 1.0,
                 delay: float = 0.001,
                 plasticity_enabled: bool = True):
        
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.weight = weight
        self.delay = delay
        self.plasticity_enabled = plasticity_enabled
        
        # Spike-timing-dependent plasticity (STDP) parameters
        self.stdp_tau_plus = 0.020  # 20ms
        self.stdp_tau_minus = 0.020  # 20ms
        self.stdp_a_plus = 0.01
        self.stdp_a_minus = 0.012
        
        # Spike timing history for STDP
        self.pre_spike_times = []
        self.post_spike_times = []
        
    def process_spike(self, spike: Spike, spike_time: float) -> float:
        """Process incoming spike and return synaptic current."""
        if spike.neuron_id == self.pre_neuron_id:
            # Pre-synaptic spike
            self.pre_spike_times.append(spike_time)
            
            # Apply STDP if plasticity enabled
            if self.plasticity_enabled:
                self._apply_stdp(spike_time, is_pre=True)
            
            # Return synaptic current (will be delayed)
            return self.weight * spike.amplitude
        
        elif spike.neuron_id == self.post_neuron_id:
            # Post-synaptic spike
            self.post_spike_times.append(spike_time)
            
            # Apply STDP if plasticity enabled
            if self.plasticity_enabled:
                self._apply_stdp(spike_time, is_pre=False)
        
        return 0.0
    
    def _apply_stdp(self, spike_time: float, is_pre: bool):
        """Apply spike-timing-dependent plasticity."""
        if is_pre:
            # Pre-synaptic spike - look for recent post-synaptic spikes
            for post_time in self.post_spike_times:
                dt = post_time - spike_time  # dt > 0 for causal pairing
                
                if abs(dt) < 0.1:  # Within STDP window
                    if dt > 0:  # Causal: pre before post
                        weight_change = self.stdp_a_plus * np.exp(-dt / self.stdp_tau_plus)
                    else:  # Anti-causal: post before pre
                        weight_change = -self.stdp_a_minus * np.exp(dt / self.stdp_tau_minus)
                    
                    self.weight += weight_change
                    self.weight = max(0.0, min(2.0, self.weight))  # Bounded weights
        
        else:
            # Post-synaptic spike - look for recent pre-synaptic spikes
            for pre_time in self.pre_spike_times:
                dt = spike_time - pre_time  # dt > 0 for causal pairing
                
                if abs(dt) < 0.1:  # Within STDP window
                    if dt > 0:  # Causal: pre before post
                        weight_change = self.stdp_a_plus * np.exp(-dt / self.stdp_tau_plus)
                    else:  # Anti-causal: post before pre
                        weight_change = -self.stdp_a_minus * np.exp(dt / self.stdp_tau_minus)
                    
                    self.weight += weight_change
                    self.weight = max(0.0, min(2.0, self.weight))  # Bounded weights
        
        # Cleanup old spike times (keep only recent)
        current_time = time.time()
        self.pre_spike_times = [t for t in self.pre_spike_times if current_time - t < 0.2]
        self.post_spike_times = [t for t in self.post_spike_times if current_time - t < 0.2]

class NeuromorphicNetwork:
    """Neuromorphic neural network for swarm processing."""
    
    def __init__(self, network_config: Dict[str, Any]):
        self.neurons = {}
        self.synapses = {}
        self.input_neurons = []
        self.output_neurons = []
        
        # Network timing
        self.dt = network_config.get('dt', 0.001)  # 1ms time step
        self.simulation_time = 0.0
        
        # Event queue for spike processing
        self.spike_queue = []
        self.processing_thread = None
        self.running = False
        
        # Performance metrics
        self.energy_consumption = 0.0
        self.spike_count = 0
        
        self._build_network(network_config)
    
    def _build_network(self, config: Dict[str, Any]):
        """Build the neuromorphic network topology."""
        # Create neurons
        num_input = config.get('num_input', 10)
        num_hidden = config.get('num_hidden', 50)
        num_output = config.get('num_output', 5)
        
        neuron_id = 0
        
        # Input layer
        for i in range(num_input):
            neuron = SpikingNeuron(neuron_id)
            self.neurons[neuron_id] = neuron
            self.input_neurons.append(neuron_id)
            neuron_id += 1
        
        # Hidden layer
        hidden_neurons = []
        for i in range(num_hidden):
            neuron = SpikingNeuron(neuron_id)
            self.neurons[neuron_id] = neuron
            hidden_neurons.append(neuron_id)
            neuron_id += 1
        
        # Output layer
        for i in range(num_output):
            neuron = SpikingNeuron(neuron_id)
            self.neurons[neuron_id] = neuron
            self.output_neurons.append(neuron_id)
            neuron_id += 1
        
        # Create synaptic connections
        synapse_id = 0
        
        # Input to hidden connections
        for input_id in self.input_neurons:
            for hidden_id in hidden_neurons:
                weight = np.random.normal(0.5, 0.2)  # Random initial weights
                synapse = SynapticConnection(input_id, hidden_id, weight)
                self.synapses[synapse_id] = synapse
                synapse_id += 1
        
        # Hidden to output connections
        for hidden_id in hidden_neurons:
            for output_id in self.output_neurons:
                weight = np.random.normal(0.5, 0.2)
                synapse = SynapticConnection(hidden_id, output_id, weight)
                self.synapses[synapse_id] = synapse
                synapse_id += 1
        
        # Recurrent connections in hidden layer (sparse)
        for i, hidden_id1 in enumerate(hidden_neurons):
            for j, hidden_id2 in enumerate(hidden_neurons):
                if i != j and np.random.random() < 0.1:  # 10% connectivity
                    weight = np.random.normal(0.2, 0.1)
                    synapse = SynapticConnection(hidden_id1, hidden_id2, weight)
                    self.synapses[synapse_id] = synapse
                    synapse_id += 1
    
    def encode_input(self, data: np.ndarray) -> List[float]:
        """Encode input data as spike rates."""
        # Normalize data to spike rates (0-100 Hz)
        normalized_data = np.clip(data, 0, 1) * 100.0
        
        # Convert to Poisson spike trains
        spike_rates = []
        for i, rate in enumerate(normalized_data[:len(self.input_neurons)]):
            spike_rates.append(rate)
        
        # Pad if necessary
        while len(spike_rates) < len(self.input_neurons):
            spike_rates.append(0.0)
        
        return spike_rates
    
    def inject_input_spikes(self, spike_rates: List[float]):
        """Inject input spikes based on rates."""
        current_time = time.time()
        
        for i, (neuron_id, rate) in enumerate(zip(self.input_neurons, spike_rates)):
            # Generate Poisson spikes
            if rate > 0 and np.random.random() < rate * self.dt / 1000.0:
                # Inject current to trigger spike
                self.neurons[neuron_id].inject_current(2.0)  # Strong current
    
    def simulate_step(self) -> Dict[int, Optional[Spike]]:
        """Simulate one time step of the network."""
        self.simulation_time += self.dt
        current_spikes = {}
        
        # Update all neurons
        for neuron_id, neuron in self.neurons.items():
            spike = neuron.update(self.dt, neuron.input_current)
            if spike:
                current_spikes[neuron_id] = spike
                self.spike_count += 1
                
                # Calculate energy consumption (simplified)
                self.energy_consumption += 0.001  # 1 nJ per spike
            
            # Reset input current
            neuron.input_current = 0.0
        
        # Process synaptic transmission
        for synapse in self.synapses.values():
            for neuron_id, spike in current_spikes.items():
                synaptic_current = synapse.process_spike(spike, self.simulation_time)
                
                # Apply synaptic current to post-synaptic neuron
                if spike.neuron_id == synapse.pre_neuron_id:
                    post_neuron = self.neurons[synapse.post_neuron_id]
                    post_neuron.input_current += synaptic_current
        
        return current_spikes
    
    def process_swarm_data(self, 
                          agent_data: Dict[int, np.ndarray],
                          duration: float = 0.1) -> Dict[str, Any]:
        """Process swarm agent data through neuromorphic network."""
        # Flatten and concatenate agent data
        all_data = []
        for agent_id, data in agent_data.items():
            all_data.extend(data.flatten()[:2])  # Take first 2 elements
        
        # Normalize to network input size
        input_data = np.array(all_data[:len(self.input_neurons)])
        if len(input_data) < len(self.input_neurons):
            input_data = np.pad(input_data, (0, len(self.input_neurons) - len(input_data)))
        
        # Encode as spike rates
        spike_rates = self.encode_input(input_data)
        
        # Simulate network
        num_steps = int(duration / self.dt)
        output_spikes = []
        
        for step in range(num_steps):
            # Inject input
            self.inject_input_spikes(spike_rates)
            
            # Simulate step
            spikes = self.simulate_step()
            
            # Collect output spikes
            step_output = []
            for output_id in self.output_neurons:
                if output_id in spikes:
                    step_output.append(1.0)
                else:
                    step_output.append(0.0)
            
            output_spikes.append(step_output)
        
        # Decode output
        output_rates = np.mean(output_spikes, axis=0) / self.dt * 1000.0  # Convert to Hz
        
        return {
            'output_rates': output_rates,
            'total_spikes': self.spike_count,
            'energy_consumption': self.energy_consumption,
            'processing_time': duration,
            'efficiency': len(agent_data) / (self.energy_consumption + 1e-9)  # agents/nJ
        }

class NeuromorphicSwarmProcessor:
    """High-level neuromorphic processor for swarm intelligence."""
    
    def __init__(self, 
                 max_agents: int = 1000,
                 processing_cores: int = 4):
        self.max_agents = max_agents
        self.processing_cores = processing_cores
        
        # Create multiple neuromorphic networks for parallel processing
        self.networks = {}
        for core_id in range(processing_cores):
            network_config = {
                'num_input': 20,
                'num_hidden': 100,
                'num_output': 10,
                'dt': 0.001
            }
            self.networks[core_id] = NeuromorphicNetwork(network_config)
        
        # Performance tracking
        self.total_energy = 0.0
        self.total_operations = 0
        self.processing_history = []
        
    def process_swarm_behavior(self,
                             agent_positions: Dict[int, np.ndarray],
                             agent_velocities: Dict[int, np.ndarray],
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process swarm behavior using neuromorphic computing.
        
        Returns behavioral analysis and control signals.
        """
        start_time = time.time()
        
        # Combine position and velocity data
        agent_data = {}
        for agent_id in agent_positions.keys():
            position = agent_positions.get(agent_id, np.array([0.0, 0.0]))
            velocity = agent_velocities.get(agent_id, np.array([0.0, 0.0]))
            agent_data[agent_id] = np.concatenate([position, velocity])
        
        # Distribute agents across processing cores
        agent_groups = self._distribute_agents(agent_data)
        
        # Process each group in parallel (simulated)
        core_results = {}
        total_energy = 0.0
        
        for core_id, group_data in agent_groups.items():
            if core_id in self.networks:
                result = self.networks[core_id].process_swarm_data(group_data)
                core_results[core_id] = result
                total_energy += result['energy_consumption']
        
        # Aggregate results
        aggregated_result = self._aggregate_core_results(core_results)
        
        # Add timing and efficiency metrics
        processing_time = time.time() - start_time
        
        # Calculate neuromorphic advantages
        advantages = self._calculate_neuromorphic_advantages(
            len(agent_data), 
            total_energy, 
            processing_time
        )
        
        self.total_energy += total_energy
        self.total_operations += 1
        
        final_result = {
            'behavioral_analysis': aggregated_result,
            'energy_consumption': total_energy,
            'processing_time': processing_time,
            'agents_processed': len(agent_data),
            'neuromorphic_advantages': advantages,
            'efficiency_metrics': {
                'energy_per_agent': total_energy / max(len(agent_data), 1),
                'agents_per_second': len(agent_data) / processing_time,
                'energy_efficiency': len(agent_data) / (total_energy + 1e-9)
            }
        }
        
        self.processing_history.append(final_result)
        return final_result
    
    def _distribute_agents(self, 
                          agent_data: Dict[int, np.ndarray]) -> Dict[int, Dict[int, np.ndarray]]:
        """Distribute agents across processing cores."""
        agent_groups = {core_id: {} for core_id in range(self.processing_cores)}
        
        agent_ids = list(agent_data.keys())
        
        for i, agent_id in enumerate(agent_ids):
            core_id = i % self.processing_cores
            agent_groups[core_id][agent_id] = agent_data[agent_id]
        
        return agent_groups
    
    def _aggregate_core_results(self, 
                              core_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple processing cores."""
        if not core_results:
            return {}
        
        # Aggregate output rates
        all_output_rates = []
        total_spikes = 0
        
        for result in core_results.values():
            all_output_rates.append(result['output_rates'])
            total_spikes += result['total_spikes']
        
        # Calculate overall behavior classification
        avg_output_rates = np.mean(all_output_rates, axis=0)
        
        # Classify behavior based on output patterns
        behavior_classification = self._classify_swarm_behavior(avg_output_rates)
        
        return {
            'output_rates': avg_output_rates,
            'total_spikes': total_spikes,
            'behavior_classification': behavior_classification,
            'core_count': len(core_results)
        }
    
    def _classify_swarm_behavior(self, output_rates: np.ndarray) -> Dict[str, float]:
        """Classify swarm behavior based on neuromorphic output."""
        if len(output_rates) < 5:
            return {'unknown': 1.0}
        
        # Simple classification based on output patterns
        classifications = {
            'flocking': output_rates[0] / 100.0,
            'dispersing': output_rates[1] / 100.0,
            'clustering': output_rates[2] / 100.0,
            'searching': output_rates[3] / 100.0,
            'coordinated': output_rates[4] / 100.0
        }
        
        # Normalize to probabilities
        total = sum(classifications.values())
        if total > 0:
            classifications = {k: v/total for k, v in classifications.items()}
        
        return classifications
    
    def _calculate_neuromorphic_advantages(self,
                                         num_agents: int,
                                         energy_consumed: float,
                                         processing_time: float) -> Dict[str, float]:
        """Calculate advantages of neuromorphic processing."""
        
        # Estimate equivalent classical processing requirements
        classical_energy = num_agents * 0.01  # 10mJ per agent (estimated)
        classical_time = num_agents * 0.001   # 1ms per agent (estimated)
        
        # Calculate advantages
        advantages = {
            'energy_efficiency': classical_energy / (energy_consumed + 1e-9),
            'speed_advantage': classical_time / (processing_time + 1e-9),
            'power_efficiency': (classical_energy / classical_time) / ((energy_consumed + 1e-9) / (processing_time + 1e-9)),
            'scalability_factor': min(2.0, 1.0 + np.log10(num_agents) * 0.1)
        }
        
        return advantages
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate neuromorphic processing performance report."""
        if not self.processing_history:
            return {}
        
        recent_results = self.processing_history[-10:]  # Last 10 operations
        
        avg_energy = np.mean([r['energy_consumption'] for r in recent_results])
        avg_processing_time = np.mean([r['processing_time'] for r in recent_results])
        avg_agents = np.mean([r['agents_processed'] for r in recent_results])
        
        return {
            'total_operations': self.total_operations,
            'total_energy_consumed': self.total_energy,
            'average_energy_per_operation': avg_energy,
            'average_processing_time': avg_processing_time,
            'average_agents_processed': avg_agents,
            'energy_efficiency_trend': [r['efficiency_metrics']['energy_efficiency'] for r in recent_results],
            'neuromorphic_advantages': recent_results[-1]['neuromorphic_advantages'] if recent_results else {},
            'network_statistics': {
                core_id: {
                    'total_spikes': network.spike_count,
                    'energy_consumption': network.energy_consumption
                }
                for core_id, network in self.networks.items()
            }
        }