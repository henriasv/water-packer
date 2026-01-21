"""Spatial hash grid for O(1) collision detection with periodic boundaries."""

import numpy as np
from typing import Tuple, Iterator, List


class SpatialHashGrid:
    """
    Spatial hash grid for efficient O(1) collision detection.
    
    Uses a dictionary-based grid to store positions and check collisions
    only against nearby atoms (27 neighboring cells).
    
    Parameters
    ----------
    cell : np.ndarray
        (3, 3) cell matrix.
    grid_spacing : float
        Size of each grid cell in Angstrom. Should be >= max collision distance.
    """
    
    def __init__(self, cell: np.ndarray, grid_spacing: float = 3.0):
        self.cell = cell
        self.inv_cell = np.linalg.inv(cell)
        self.grid_spacing = grid_spacing
        
        # Calculate number of grid cells in each direction
        cell_lengths = np.linalg.norm(cell, axis=1)
        self.n_cells = np.maximum((cell_lengths / grid_spacing).astype(int), 1)
        
        # Storage: {(i,j,k): [(pos, species), ...]}
        self.grid = {}
    
    def _get_cell_index(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """Convert Cartesian position to grid cell index."""
        # Convert to fractional coordinates
        frac = np.dot(pos, self.inv_cell)
        # Scale to grid and wrap
        idx = (frac * self.n_cells).astype(int) % self.n_cells
        return tuple(idx)
    
    def _get_neighbor_cells(self, idx: Tuple[int, int, int]) -> Iterator[Tuple[int, int, int]]:
        """
        Get all 27 neighboring cells (including self).
        
        Handles periodic boundary conditions.
        """
        idx_arr = np.array(idx)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    neighbor = tuple((idx_arr + [di, dj, dk]) % self.n_cells)
                    yield neighbor
    
    def _periodic_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Calculate distance between two positions with minimum image convention.
        """
        diff = pos1 - pos2
        # Convert to fractional, apply MIC, convert back
        frac_diff = np.dot(diff, self.inv_cell)
        frac_diff -= np.round(frac_diff)
        cart_diff = np.dot(frac_diff, self.cell)
        return np.linalg.norm(cart_diff)
    
    def check_collision(
        self,
        pos: np.ndarray,
        species: str,
        min_dist_func,
    ) -> bool:
        """
        Check if position collides with any existing atoms.
        
        Parameters
        ----------
        pos : np.ndarray
            Position to check.
        species : str
            Element symbol of the atom being placed.
        min_dist_func : callable
            Function(species1, species2) -> min_distance
        
        Returns
        -------
        bool
            True if collision detected, False if safe.
        """
        idx = self._get_cell_index(pos)
        
        # Check all neighboring cells
        for neighbor_idx in self._get_neighbor_cells(idx):
            if neighbor_idx not in self.grid:
                continue
            
            # Check against all atoms in this cell
            for existing_pos, existing_species in self.grid[neighbor_idx]:
                dist = self._periodic_distance(pos, existing_pos)
                min_dist = min_dist_func(species, existing_species)
                
                if dist < min_dist:
                    return True  # Collision!
        
        return False  # Safe
    
    def add(self, pos: np.ndarray, species: str):
        """
        Add a position to the grid.
        
        Parameters
        ----------
        pos : np.ndarray
            Position to add.
        species : str
            Element symbol.
        """
        idx = self._get_cell_index(pos)
        if idx not in self.grid:
            self.grid[idx] = []
        self.grid[idx].append((pos.copy(), species))
    
    def add_multiple(self, positions: np.ndarray, species: List[str]):
        """
        Add multiple positions at once.
        
        Parameters
        ----------
        positions : np.ndarray
            (N, 3) array of positions.
        species : list of str
            Element symbols for each position.
        """
        for pos, sp in zip(positions, species):
            self.add(pos, sp)
    
    def count(self) -> int:
        """Return total number of atoms in grid."""
        return sum(len(atoms) for atoms in self.grid.values())
    
    def clear(self):
        """Clear all atoms from grid."""
        self.grid.clear()
