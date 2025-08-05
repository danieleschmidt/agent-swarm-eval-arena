"""Spatial indexing utilities for efficient neighbor queries."""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import math

from ..core.config import SwarmConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SpatialHashGrid:
    """Spatial hash grid for efficient neighbor queries in 2D space.
    
    Provides O(1) average case neighbor lookups instead of O(n²) brute force.
    Optimized for dynamic agent environments with frequent updates.
    """
    
    def __init__(self, arena_size: Tuple[float, float], cell_size: float = 50.0):
        """Initialize spatial hash grid.
        
        Args:
            arena_size: (width, height) of the arena
            cell_size: Size of each grid cell (should be ~2x max interaction radius)
        """
        self.arena_size = arena_size
        self.cell_size = cell_size
        
        # Calculate grid dimensions
        self.grid_width = int(math.ceil(arena_size[0] / cell_size))
        self.grid_height = int(math.ceil(arena_size[1] / cell_size))
        
        # Grid storage: cell_key -> set of agent_ids
        self.grid: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        
        # Agent position tracking
        self.agent_positions: Dict[int, np.ndarray] = {}
        self.agent_cells: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        
        # Performance tracking
        self.update_count = 0
        self.query_count = 0
        
        logger.info(f"Spatial grid initialized: {self.grid_width}x{self.grid_height} cells "
                   f"(cell_size={cell_size}, arena={arena_size})")
    
    def _position_to_cell(self, position: np.ndarray) -> Tuple[int, int]:
        """Convert world position to grid cell coordinates.
        
        Args:
            position: World position [x, y]
            
        Returns:
            Grid cell coordinates (col, row)
        """
        col = max(0, min(int(position[0] / self.cell_size), self.grid_width - 1))
        row = max(0, min(int(position[1] / self.cell_size), self.grid_height - 1))
        return (col, row)
    
    def _get_neighbor_cells(self, cell: Tuple[int, int], radius: int = 1) -> List[Tuple[int, int]]:
        """Get neighboring cells within radius.
        
        Args:
            cell: Center cell (col, row)
            radius: Search radius in cells
            
        Returns:
            List of valid neighboring cell coordinates
        """
        col, row = cell
        neighbors = []
        
        for dc in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                new_col = col + dc
                new_row = row + dr
                
                # Check bounds
                if (0 <= new_col < self.grid_width and 
                    0 <= new_row < self.grid_height):
                    neighbors.append((new_col, new_row))
        
        return neighbors
    
    def update_agent(self, agent_id: int, position: np.ndarray) -> None:
        """Update agent position in spatial grid.
        
        Args:
            agent_id: Unique agent identifier
            position: New position [x, y]
        """
        # Remove from old cells
        if agent_id in self.agent_cells:
            for old_cell in self.agent_cells[agent_id]:
                self.grid[old_cell].discard(agent_id)
                # Clean up empty cells
                if not self.grid[old_cell]:
                    del self.grid[old_cell]
        
        # Calculate new cell
        new_cell = self._position_to_cell(position)
        
        # Update tracking
        self.agent_positions[agent_id] = position.copy()
        self.agent_cells[agent_id] = {new_cell}
        
        # Add to new cell
        self.grid[new_cell].add(agent_id)
        
        self.update_count += 1
    
    def remove_agent(self, agent_id: int) -> None:
        """Remove agent from spatial grid.
        
        Args:
            agent_id: Agent to remove
        """
        if agent_id not in self.agent_cells:
            return
        
        # Remove from all cells
        for cell in self.agent_cells[agent_id]:
            self.grid[cell].discard(agent_id)
            if not self.grid[cell]:
                del self.grid[cell]
        
        # Clean up tracking
        del self.agent_cells[agent_id]
        self.agent_positions.pop(agent_id, None)
    
    def query_neighbors(self, position: np.ndarray, radius: float, 
                       exclude_id: Optional[int] = None) -> List[Tuple[int, float]]:
        """Find all agents within radius of position.
        
        Args:
            position: Query position [x, y]
            radius: Search radius
            exclude_id: Agent ID to exclude from results
            
        Returns:
            List of (agent_id, distance) tuples within radius
        """
        self.query_count += 1
        
        # Calculate search radius in cells
        cell_radius = max(1, int(math.ceil(radius / self.cell_size)))
        query_cell = self._position_to_cell(position)
        
        # Get candidate cells
        neighbor_cells = self._get_neighbor_cells(query_cell, cell_radius)
        
        # Collect candidates from cells
        candidates = set()
        for cell in neighbor_cells:
            if cell in self.grid:
                candidates.update(self.grid[cell])
        
        # Filter by actual distance
        results = []
        for agent_id in candidates:
            if exclude_id is not None and agent_id == exclude_id:
                continue
                
            if agent_id not in self.agent_positions:
                continue
            
            agent_pos = self.agent_positions[agent_id]
            distance = np.linalg.norm(position - agent_pos)
            
            if distance <= radius:
                results.append((agent_id, distance))
        
        # Sort by distance
        results.sort(key=lambda x: x[1])
        return results
    
    def query_neighbors_fast(self, agent_id: int, radius: float) -> List[int]:
        """Fast neighbor query for existing agent (returns IDs only).
        
        Args:
            agent_id: Agent making the query
            radius: Search radius
            
        Returns:
            List of neighbor agent IDs within radius
        """
        if agent_id not in self.agent_positions:
            return []
        
        position = self.agent_positions[agent_id]
        neighbors = self.query_neighbors(position, radius, exclude_id=agent_id)
        return [neighbor_id for neighbor_id, _ in neighbors]
    
    def get_agent_count_in_cell(self, position: np.ndarray) -> int:
        """Get number of agents in the same cell as position.
        
        Args:
            position: Query position
            
        Returns:
            Number of agents in cell
        """
        cell = self._position_to_cell(position)
        return len(self.grid.get(cell, set()))
    
    def get_density_map(self) -> Dict[Tuple[int, int], int]:
        """Get agent density for each occupied cell.
        
        Returns:
            Dictionary mapping cell coordinates to agent count
        """
        return {cell: len(agents) for cell, agents in self.grid.items()}
    
    def get_statistics(self) -> Dict[str, any]:
        """Get spatial grid statistics.
        
        Returns:
            Dictionary of performance and usage statistics
        """
        occupied_cells = len(self.grid)
        total_cells = self.grid_width * self.grid_height
        agent_count = len(self.agent_positions)
        
        # Calculate average agents per occupied cell
        avg_agents_per_cell = (sum(len(agents) for agents in self.grid.values()) / 
                              occupied_cells if occupied_cells > 0 else 0)
        
        return {
            "total_cells": total_cells,
            "occupied_cells": occupied_cells,
            "occupancy_ratio": occupied_cells / total_cells,
            "agent_count": agent_count,
            "avg_agents_per_cell": avg_agents_per_cell,
            "cell_size": self.cell_size,
            "grid_dimensions": (self.grid_width, self.grid_height),
            "update_count": self.update_count,
            "query_count": self.query_count,
            "queries_per_update": self.query_count / max(1, self.update_count)
        }
    
    def clear(self) -> None:
        """Clear all data from spatial grid."""
        self.grid.clear()
        self.agent_positions.clear()
        self.agent_cells.clear()
        self.update_count = 0
        self.query_count = 0


class SpatialQuadTree:
    """Quadtree implementation for hierarchical spatial indexing.
    
    Better than hash grid for non-uniform agent distributions.
    Provides adaptive subdivision based on agent density.
    """
    
    def __init__(self, bounds: Tuple[float, float, float, float], 
                 max_agents: int = 10, max_depth: int = 8):
        """Initialize quadtree.
        
        Args:
            bounds: (min_x, min_y, max_x, max_y) bounding rectangle
            max_agents: Maximum agents per leaf before subdivision
            max_depth: Maximum tree depth
        """
        self.bounds = bounds  # (min_x, min_y, max_x, max_y)
        self.max_agents = max_agents
        self.max_depth = max_depth
        self.depth = 0
        
        # Node data
        self.agents: Dict[int, np.ndarray] = {}  # agent_id -> position
        self.children: Optional[List['SpatialQuadTree']] = None
        self.is_leaf = True
        
        # Performance tracking
        self.insert_count = 0
        self.query_count = 0
    
    def _should_subdivide(self) -> bool:
        """Check if node should be subdivided."""
        return (self.is_leaf and 
                len(self.agents) > self.max_agents and 
                self.depth < self.max_depth)
    
    def _subdivide(self) -> None:
        """Subdivide current node into four children."""
        if not self.is_leaf:
            return
        
        min_x, min_y, max_x, max_y = self.bounds
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        
        # Create four child quadrants
        self.children = [
            SpatialQuadTree((min_x, min_y, mid_x, mid_y), 
                           self.max_agents, self.max_depth),  # SW
            SpatialQuadTree((mid_x, min_y, max_x, mid_y), 
                           self.max_agents, self.max_depth),  # SE
            SpatialQuadTree((min_x, mid_y, mid_x, max_y), 
                           self.max_agents, self.max_depth),  # NW
            SpatialQuadTree((mid_x, mid_y, max_x, max_y), 
                           self.max_agents, self.max_depth),  # NE
        ]
        
        # Set child depths
        for child in self.children:
            child.depth = self.depth + 1
        
        # Redistribute agents to children
        agents_to_redistribute = self.agents.copy()
        self.agents.clear()
        
        for agent_id, position in agents_to_redistribute.items():
            self._insert_into_child(agent_id, position)
        
        self.is_leaf = False
    
    def _get_quadrant(self, position: np.ndarray) -> int:
        """Get quadrant index for position.
        
        Args:
            position: Position [x, y]
            
        Returns:
            Quadrant index (0=SW, 1=SE, 2=NW, 3=NE)
        """
        min_x, min_y, max_x, max_y = self.bounds
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        
        if position[0] < mid_x:
            return 2 if position[1] >= mid_y else 0  # NW or SW
        else:
            return 3 if position[1] >= mid_y else 1  # NE or SE
    
    def _insert_into_child(self, agent_id: int, position: np.ndarray) -> None:
        """Insert agent into appropriate child."""
        if self.children is None:
            return
        
        quadrant = self._get_quadrant(position)
        self.children[quadrant].insert(agent_id, position)
    
    def _contains_point(self, position: np.ndarray) -> bool:
        """Check if position is within bounds."""
        min_x, min_y, max_x, max_y = self.bounds
        return (min_x <= position[0] <= max_x and 
                min_y <= position[1] <= max_y)
    
    def _intersects_circle(self, center: np.ndarray, radius: float) -> bool:
        """Check if bounds intersect with circle."""
        min_x, min_y, max_x, max_y = self.bounds
        
        # Find closest point on rectangle to circle center
        closest_x = max(min_x, min(center[0], max_x))
        closest_y = max(min_y, min(center[1], max_y))
        
        # Check if closest point is within circle
        distance = np.linalg.norm([closest_x - center[0], closest_y - center[1]])
        return distance <= radius
    
    def insert(self, agent_id: int, position: np.ndarray) -> bool:
        """Insert agent at position.
        
        Args:
            agent_id: Unique agent identifier
            position: Agent position [x, y]
            
        Returns:
            True if inserted successfully
        """
        if not self._contains_point(position):
            return False
        
        self.insert_count += 1
        
        if self.is_leaf:
            self.agents[agent_id] = position.copy()
            
            # Check if subdivision is needed
            if self._should_subdivide():
                self._subdivide()
        else:
            # Insert into appropriate child
            self._insert_into_child(agent_id, position)
        
        return True
    
    def remove(self, agent_id: int) -> bool:
        """Remove agent from tree.
        
        Args:
            agent_id: Agent to remove
            
        Returns:
            True if removed successfully
        """
        if self.is_leaf:
            return agent_id in self.agents and self.agents.pop(agent_id, None) is not None
        else:
            # Try removing from children
            if self.children:
                for child in self.children:
                    if child.remove(agent_id):
                        return True
        return False
    
    def query_range(self, center: np.ndarray, radius: float, 
                   exclude_id: Optional[int] = None) -> List[Tuple[int, float]]:
        """Query agents within radius of center.
        
        Args:
            center: Query center [x, y]
            radius: Search radius
            exclude_id: Agent ID to exclude
            
        Returns:
            List of (agent_id, distance) tuples
        """
        self.query_count += 1
        results = []
        
        # Check if this node intersects with query circle
        if not self._intersects_circle(center, radius):
            return results
        
        if self.is_leaf:
            # Check all agents in this leaf
            for agent_id, position in self.agents.items():
                if exclude_id is not None and agent_id == exclude_id:
                    continue
                
                distance = np.linalg.norm(center - position)
                if distance <= radius:
                    results.append((agent_id, distance))
        else:
            # Query children
            if self.children:
                for child in self.children:
                    results.extend(child.query_range(center, radius, exclude_id))
        
        return results
    
    def get_statistics(self) -> Dict[str, any]:
        """Get quadtree statistics."""
        stats = {
            "depth": self.depth,
            "is_leaf": self.is_leaf,
            "agent_count": len(self.agents),
            "insert_count": self.insert_count,
            "query_count": self.query_count,
            "bounds": self.bounds
        }
        
        if not self.is_leaf and self.children:
            child_stats = [child.get_statistics() for child in self.children]
            stats["children"] = child_stats
            stats["total_nodes"] = 1 + sum(cs.get("total_nodes", 1) for cs in child_stats)
            stats["max_depth"] = max(cs.get("max_depth", cs["depth"]) for cs in child_stats)
        else:
            stats["total_nodes"] = 1
            stats["max_depth"] = self.depth
        
        return stats
    
    def clear(self) -> None:
        """Clear all data from quadtree."""
        self.agents.clear()
        self.children = None
        self.is_leaf = True
        self.insert_count = 0
        self.query_count = 0


def create_spatial_index(config: SwarmConfig, index_type: str = "hash_grid") -> any:
    """Factory function to create spatial index based on configuration.
    
    Args:
        config: Swarm configuration
        index_type: Type of spatial index ("hash_grid" or "quadtree")
        
    Returns:
        Spatial index instance
    """
    if index_type == "hash_grid":
        # Cell size should be ~2x the observation radius for optimal performance
        cell_size = min(config.observation_radius * 2.0, 100.0)
        return SpatialHashGrid(config.arena_size, cell_size)
    
    elif index_type == "quadtree":
        bounds = (0, 0, config.arena_size[0], config.arena_size[1])
        max_agents = min(20, config.num_agents // 10)  # Adaptive based on total agents
        return SpatialQuadTree(bounds, max_agents)
    
    else:
        raise ValueError(f"Unknown spatial index type: {index_type}")