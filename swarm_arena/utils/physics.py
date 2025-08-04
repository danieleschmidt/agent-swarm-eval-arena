"""Physics engines for the Swarm Arena."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np
from ..core.config import SwarmConfig


class PhysicsEngine(ABC):
    """Abstract base class for physics engines."""
    
    def __init__(self, config: SwarmConfig) -> None:
        """Initialize physics engine.
        
        Args:
            config: Swarm configuration
        """
        self.config = config
    
    @abstractmethod
    def step(self, positions: Dict[int, np.ndarray], velocities: Dict[int, np.ndarray], 
             dt: float) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Execute one physics step.
        
        Args:
            positions: Current agent positions  
            velocities: Current agent velocities
            dt: Time step
            
        Returns:
            Tuple of (new_positions, new_velocities)
        """
        pass
    
    @abstractmethod
    def detect_collisions(self, positions: Dict[int, np.ndarray]) -> List[Tuple[int, int]]:
        """Detect collisions between agents.
        
        Args:
            positions: Current agent positions
            
        Returns:
            List of (agent_id_1, agent_id_2) collision pairs
        """
        pass


class SimplePhysicsEngine(PhysicsEngine):
    """Simple 2D physics engine with basic collision detection."""
    
    def __init__(self, config: SwarmConfig) -> None:
        super().__init__(config)
        self.damping = 0.95  # Velocity damping factor
        self.collision_radius = config.collision_radius
        
    def step(self, positions: Dict[int, np.ndarray], velocities: Dict[int, np.ndarray], 
             dt: float) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Execute simple physics step with Euler integration.
        
        Args:
            positions: Current agent positions
            velocities: Current agent velocities  
            dt: Time step
            
        Returns:
            Updated positions and velocities
        """
        new_positions = {}
        new_velocities = {}
        
        for agent_id in positions.keys():
            # Simple Euler integration
            new_pos = positions[agent_id] + velocities[agent_id] * dt
            new_vel = velocities[agent_id] * self.damping
            
            # Boundary collision handling
            new_pos, new_vel = self._handle_boundary_collision(new_pos, new_vel)
            
            new_positions[agent_id] = new_pos
            new_velocities[agent_id] = new_vel
        
        return new_positions, new_velocities
    
    def detect_collisions(self, positions: Dict[int, np.ndarray]) -> List[Tuple[int, int]]:
        """Detect agent-agent collisions using distance threshold.
        
        Args:
            positions: Current agent positions
            
        Returns:
            List of collision pairs
        """
        collisions = []
        agent_ids = list(positions.keys())
        
        for i, agent_id_1 in enumerate(agent_ids):
            for agent_id_2 in agent_ids[i+1:]:
                distance = np.linalg.norm(positions[agent_id_1] - positions[agent_id_2])
                
                if distance < self.collision_radius:
                    collisions.append((agent_id_1, agent_id_2))
        
        return collisions
    
    def resolve_collision(self, pos1: np.ndarray, pos2: np.ndarray, 
                         vel1: np.ndarray, vel2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve collision between two agents using elastic collision.
        
        Args:
            pos1, pos2: Agent positions
            vel1, vel2: Agent velocities
            
        Returns:
            Updated velocities after collision
        """
        # Calculate collision normal
        diff = pos2 - pos1
        distance = np.linalg.norm(diff)
        
        if distance < 1e-6:  # Avoid division by zero
            normal = np.array([1.0, 0.0])
        else:
            normal = diff / distance
        
        # Relative velocity
        rel_vel = vel2 - vel1
        
        # Relative velocity in collision normal direction
        vel_normal = np.dot(rel_vel, normal)
        
        # Don't resolve if velocities are separating
        if vel_normal > 0:
            return vel1, vel2
        
        # Elastic collision (equal masses)
        impulse = 2 * vel_normal / 2  # Assuming equal masses = 1
        
        new_vel1 = vel1 + impulse * normal
        new_vel2 = vel2 - impulse * normal
        
        return new_vel1, new_vel2
    
    def _handle_boundary_collision(self, position: np.ndarray, 
                                  velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle collision with arena boundaries.
        
        Args:
            position: Agent position
            velocity: Agent velocity
            
        Returns:
            Updated position and velocity
        """
        new_pos = position.copy()
        new_vel = velocity.copy()
        
        # X boundaries
        if new_pos[0] < 0:
            new_pos[0] = 0
            new_vel[0] = -new_vel[0] * 0.8  # Some energy loss
        elif new_pos[0] > self.config.arena_size[0]:
            new_pos[0] = self.config.arena_size[0]
            new_vel[0] = -new_vel[0] * 0.8
        
        # Y boundaries  
        if new_pos[1] < 0:
            new_pos[1] = 0
            new_vel[1] = -new_vel[1] * 0.8
        elif new_pos[1] > self.config.arena_size[1]:
            new_pos[1] = self.config.arena_size[1]
            new_vel[1] = -new_vel[1] * 0.8
        
        return new_pos, new_vel


class MujocoPhysicsEngine(PhysicsEngine):
    """MuJoCo-based physics engine for more realistic simulations."""
    
    def __init__(self, config: SwarmConfig) -> None:
        super().__init__(config)
        # Placeholder for MuJoCo integration
        # This would require actual MuJoCo installation and setup
        self.mujoco_available = False
        
        if not self.mujoco_available:
            print("Warning: MuJoCo not available, falling back to SimplePhysicsEngine")
            self.fallback_engine = SimplePhysicsEngine(config)
    
    def step(self, positions: Dict[int, np.ndarray], velocities: Dict[int, np.ndarray], 
             dt: float) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Execute MuJoCo physics step."""
        if not self.mujoco_available:
            return self.fallback_engine.step(positions, velocities, dt)
        
        # MuJoCo physics implementation would go here
        raise NotImplementedError("MuJoCo integration not yet implemented")
    
    def detect_collisions(self, positions: Dict[int, np.ndarray]) -> List[Tuple[int, int]]:
        """Detect collisions using MuJoCo."""
        if not self.mujoco_available:
            return self.fallback_engine.detect_collisions(positions)
        
        raise NotImplementedError("MuJoCo integration not yet implemented")