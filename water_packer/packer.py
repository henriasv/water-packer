"""
Optimized Water Packer using vectorized operations and spatial hash grid.

This version uses:
- Monte Carlo volume estimation (avoids overlap errors)
- cKDTree for vectorized substrate culling (50k candidates/batch)
- Spatial hash grid for O(1) collision checking
- 10-100x faster for large systems
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .constants import WATER_DENSITY_FACTOR
from .converters import (
    detect_input_type,
    from_arrays,
    from_ase,
    from_hyobj,
    to_arrays,
    to_ase,
    to_hyobj,
)
from .distance_manager import DistanceManager
from .water_geometry import get_water_template, rotate_water
from .volume_estimation import estimate_available_volume_mc, generate_candidate_batch
from .spatial_hash import SpatialHashGrid


class WaterPackerOptimized:
    """
    High-performance water packer using vectorized operations.
    
    Uses batch generation + spatial hash grid for O(1) collision checking.
    10-100× faster than basic implementation for large systems.
    
    Parameters
    ----------
    min_distance : float
        Default minimum distance between any atoms (Å).
    pairwise_distances : dict, optional
        Species-specific minimum distances.
    water_density : float
        Target water density in g/cm³.
    seed : int, optional
        Random seed for reproducibility.
    batch_size : int
        Number of candidates to generate per batch (default 50000).
    grid_spacing : float
        Spatial hash grid cell size in Å (default 3.0).
    n_mc_probes : int
        Number of Monte Carlo probes for volume estimation (default 100000).
    """
    
    def __init__(
        self,
        min_distance: float = 2.0,
        pairwise_distances: Optional[Dict[Tuple[str, str], float]] = None,
        water_density: float = 1.0,
        seed: Optional[int] = None,
        batch_size: int = 50000,
        grid_spacing: float = 3.0,
        n_mc_probes: int = 100000,
    ):
        self.distance_manager = DistanceManager(
            default_distance=min_distance,
            pairwise_distances=pairwise_distances,
        )
        self.water_density = water_density
        self.batch_size = batch_size
        self.grid_spacing = grid_spacing
        self.n_mc_probes = n_mc_probes
        
        # Create random generator (avoids polluting global state)
        self.rng = np.random.default_rng(seed)
        
        # Water template
        self.water_symbols, self.water_template = get_water_template()
    
    def pack(
        self,
        system: Any,
        n_waters: Optional[int] = None,
        region: Optional[Dict[str, float]] = None,
    ) -> Any:
        """
        Pack water molecules into the system.
        
        Parameters
        ----------
        system : ASE Atoms, hyobj PeriodicSystem, or dict
            Input system.
        n_waters : int, optional
            Number of waters to pack. If None, calculated from density.
        region : dict, optional
            Restrict packing region.
        
        Returns
        -------
        Same type as input, with water molecules added.
        """
        # Detect and convert input
        input_type = detect_input_type(system)
        
        if input_type == 'ase':
            data = from_ase(system)
        elif input_type == 'hyobj':
            data = from_hyobj(system)
        elif input_type == 'dict':
            data = system.copy()
        else:
            raise TypeError(
                f"Unsupported system type: {type(system)}. "
                "Use ASE Atoms, hyobj PeriodicSystem, or dict."
            )
        
        # Pack water
        result_data = self._pack_water_optimized(data, n_waters, region)
        
        # Convert back to original type
        if input_type == 'ase':
            return to_ase(result_data)
        elif input_type == 'hyobj':
            return to_hyobj(result_data, units='metal')
        else:
            return result_data
    
    def _pack_water_optimized(
        self,
        data: Dict,
        n_waters: Optional[int],
        region: Optional[Dict[str, float]],
    ) -> Dict:
        """Optimized packing implementation using batch + grid."""
        cell = np.array(data['cell'])
        positions = np.array(data['positions'])
        species = list(data['species'])
        
        # Monte Carlo volume estimation
        available_volume = estimate_available_volume_mc(
            positions, species, cell,
            self.distance_manager,
            n_probes=self.n_mc_probes,
            rng=self.rng,
        )
        
        if available_volume <= 0:
            print(f"Warning: No available volume for water packing.")
            return data
        
        # Physics safeguard: Check for impossible density requests
        if n_waters is not None:
            # Calculate implied density
            # implied_conc = n_waters / available_volume (molecules/Å³)
            implied_conc = n_waters / available_volume
            
            # Theoretical max density for Random Close Packing (RCP) of spheres
            # RCP fraction ≈ 0.64
            # Sphere radius r = min_dist(O,O) / 2
            min_OO_dist = self.distance_manager.get_min_distance('O', 'O')
            r = min_OO_dist / 2.0
            sphere_vol = (4/3) * np.pi * (r**3)
            max_conc = 0.64 / sphere_vol
            
            if implied_conc > max_conc:
                implied_g_cm3 = implied_conc / WATER_DENSITY_FACTOR
                max_g_cm3 = max_conc / WATER_DENSITY_FACTOR
                print(f"\n⚠️  WARNING: Requested packing density is extremely high!")
                print(f"   - Implied density: {implied_g_cm3:.2f} g/cm³ ({n_waters} waters in {available_volume:.1f} Å³)")
                print(f"   - Theoretical max (RCP): ~{max_g_cm3:.2f} g/cm³")
                print(f"   - This will likely fail or take a very long time.\n")
                
        # Calculate number of waters if not specified
        else:
            target_concentration = self.water_density * WATER_DENSITY_FACTOR
            n_waters = int(available_volume * target_concentration)
            n_waters = max(1, n_waters)
        
        # Initialize spatial hash grid for water-water collision checking
        grid = SpatialHashGrid(cell, grid_spacing=self.grid_spacing)
        
        # Add substrate atoms to grid
        grid.add_multiple(positions, species)
        
        # Pack waters using batch generation
        placed_waters = []
        placed_species = []
        max_batch_attempts = 100  # Prevent infinite loops
        batch_attempt = 0
        
        while len(placed_waters) < n_waters and batch_attempt < max_batch_attempts:
            # Phase A: Generate batch and cull against substrate
            candidates, rotations = generate_candidate_batch(
                cell, positions, species,
                self.distance_manager,
                batch_size=self.batch_size,
                region=region,
                rng=self.rng,
            )
            
            if len(candidates) == 0:
                batch_attempt += 1
                continue
            
            # Phase B: Check grid collisions and place waters
            for i, (center_pos, rotation) in enumerate(zip(candidates, rotations)):
                if len(placed_waters) >= n_waters:
                    break
                
                # Create water molecule at this position
                # water_template is (3, 3) where rows are atoms
                # rotation is (3, 3) matrix
                # We want to rotate each atom vector: v_new = R @ v_old
                # Equivalent to: M_new = M_old @ R.T
                water_coords = self.water_template @ rotation.T + center_pos
                
                # Check collision with all atoms (substrate + waters) using grid
                has_collision = False
                for j, (atom_pos, atom_species) in enumerate(zip(water_coords, self.water_symbols)):
                    if grid.check_collision(
                        atom_pos,
                        atom_species,
                        self.distance_manager.get_min_distance,
                    ):
                        has_collision = True
                        break
                
                if not has_collision:
                    # Place water
                    placed_waters.append(water_coords)
                    placed_species.extend(self.water_symbols)
                    
                    # Add to grid for future collision checking
                    for atom_pos, atom_species in zip(water_coords, self.water_symbols):
                        grid.add(atom_pos, atom_species)
            
            batch_attempt += 1
        
        if len(placed_waters) < n_waters:
            print(f"Warning: Could only place {len(placed_waters)}/{n_waters} water molecules.")
        
        # Combine substrate + waters
        if placed_waters:
            all_positions = np.vstack([positions, np.vstack(placed_waters)])
            all_species = species + placed_species
        else:
            all_positions = positions
            all_species = species
        
        result = {
            'cell': cell,
            'positions': all_positions,
            'species': all_species,
            'pbc': data.get('pbc', True),
        }
        
        return result


# Alias for backward compatibility
WaterPacker = WaterPackerOptimized


def pack_water(
    system: Any,
    n_waters: Optional[int] = None,
    min_distance: float = 2.0,
    pairwise_distances: Optional[Dict[Tuple[str, str], float]] = None,
    water_density: Optional[float] = None,
    seed: Optional[int] = None,
) -> Any:
    """
    Convenience function to pack water into a system.
    
    Uses optimized implementation with batch generation and spatial hash grid.
    """
    if n_waters is not None and water_density is not None:
        raise ValueError("Cannot specify both 'n_waters' and 'water_density'.")
    
    if n_waters is None and water_density is None:
        water_density = 1.0
        
    packer = WaterPacker(
        min_distance=min_distance,
        pairwise_distances=pairwise_distances,
        water_density=water_density or 1.0,
        seed=seed,
    )
    return packer.pack(system, n_waters=n_waters)
