"""
Relaxation-based packing: jiggle existing waters to make room for new ones.

When standard insertion fails, perturb existing water molecules slightly
while respecting all distance constraints, then try inserting again.
"""

import numpy as np
from typing import List, Tuple, Optional


class WaterRelaxer:
    """
    Relax water positions to create space for new insertions.
    
    Uses random perturbations with constraint satisfaction.
    """
    
    def __init__(
        self,
        cell: np.ndarray,
        distance_manager,
        max_displacement: float = 0.5,
        n_relax_attempts: int = 10,
    ):
        """
        Parameters
        ----------
        cell : np.ndarray
            (3, 3) cell matrix.
        distance_manager : DistanceManager
            For distance constraints.
        max_displacement : float
            Maximum random displacement per atom (Å).
        n_relax_attempts : int
            Number of relaxation attempts per cycle.
        """
        self.cell = cell
        self.inv_cell = np.linalg.inv(cell)
        self.distance_manager = distance_manager
        self.max_displacement = max_displacement
        self.n_relax_attempts = n_relax_attempts
    
    def _wrap_positions(self, positions: np.ndarray) -> np.ndarray:
        """Wrap positions into primary cell using PBC."""
        frac = np.dot(positions, self.inv_cell)
        frac = frac % 1.0
        return np.dot(frac, self.cell)
    
    def _check_valid_position(
        self,
        pos: np.ndarray,
        species: str,
        all_positions: np.ndarray,
        all_species: List[str],
        exclude_indices: List[int],
    ) -> bool:
        """Check if position is valid (no collisions)."""
        for i, (other_pos, other_sp) in enumerate(zip(all_positions, all_species)):
            if i in exclude_indices:
                continue
            
            # Minimum image distance
            diff = pos - other_pos
            frac_diff = np.dot(diff, self.inv_cell)
            frac_diff -= np.round(frac_diff)
            dist = np.linalg.norm(np.dot(frac_diff, self.cell))
            
            min_dist = self.distance_manager.get_min_distance(species, other_sp)
            if dist < min_dist:
                return False
        
        return True
    
    def relax_waters(
        self,
        water_positions: np.ndarray,  # (N_waters, 3, 3) - each water has 3 atoms
        water_species: List[List[str]],  # [['O','H','H'], ...]
        substrate_positions: np.ndarray,
        substrate_species: List[str],
    ) -> Tuple[np.ndarray, int]:
        """
        Apply random perturbations to water molecules, accepting moves that
        maintain all distance constraints.
        
        Returns
        -------
        new_positions : np.ndarray
            Relaxed water positions.
        n_accepted : int
            Number of accepted moves.
        """
        n_waters = len(water_positions)
        if n_waters == 0:
            return water_positions, 0
        
        # Flatten for distance checking
        all_positions = np.vstack([
            substrate_positions,
            water_positions.reshape(-1, 3)
        ])
        all_species = list(substrate_species)
        for ws in water_species:
            all_species.extend(ws)
        
        n_substrate = len(substrate_positions)
        n_accepted = 0
        
        # Try to relax each water molecule
        for attempt in range(self.n_relax_attempts):
            # Pick a random water
            water_idx = np.random.randint(n_waters)
            
            # Random displacement for the whole molecule (rigid body translation)
            displacement = (np.random.random(3) - 0.5) * 2 * self.max_displacement
            
            # Optional: also random rotation
            from scipy.spatial.transform import Rotation
            small_rotation = Rotation.from_rotvec(
                (np.random.random(3) - 0.5) * 0.2  # Small angle ~0.1 rad
            ).as_matrix()
            
            # Current water atoms
            old_water = water_positions[water_idx].copy()  # (3, 3)
            center = old_water[0]  # Use O as center
            
            # Calculate relative vectors with MIC to handle PBC straddling
            # This ensures we have the correct bond vectors even if H is wrapped
            rel_vecs = old_water - center
            frac_vecs = np.dot(rel_vecs, self.inv_cell)
            frac_vecs -= np.round(frac_vecs)
            rel_vecs = np.dot(frac_vecs, self.cell)
            
            # Rotate the valid relative vectors
            new_rel_vecs = small_rotation @ rel_vecs.T
            
            # Translate center and wrap ONLY the center
            # This ensures the molecule stays contiguous in Cartesian space
            old_center = center
            new_center = old_center + displacement
            
            # Wrap new center into box
            frac_center = np.dot(new_center, self.inv_cell)
            frac_center = frac_center % 1.0
            new_center = np.dot(frac_center, self.cell)
            
            # Reconstruct molecule around wrapped center
            new_water = new_rel_vecs.T + new_center
            
            # Do NOT separately wrap atoms here - keep them contiguous
            # MIC in distance check handles the PBC checking
            
            # Check if new positions are valid
            # Indices of atoms we're moving (in all_positions)
            water_atom_start = n_substrate + water_idx * 3
            exclude_indices = list(range(water_atom_start, water_atom_start + 3))
            
            valid = True
            for i, (pos, sp) in enumerate(zip(new_water, ['O', 'H', 'H'])):
                if not self._check_valid_position(
                    pos, sp, all_positions, all_species, exclude_indices
                ):
                    valid = False
                    break
            
            if valid:
                # Accept move
                water_positions[water_idx] = new_water
                # Update all_positions
                all_positions[water_atom_start:water_atom_start+3] = new_water
                n_accepted += 1
        
        return water_positions, n_accepted


def pack_with_relaxation(
    packer,
    system,
    n_waters: int,
    relax_after_waters: int = 10,
    relax_every_failures: int = 50,
    max_cycles: int = 20,
):
    """
    Hybrid packing: insert waters, then relax when stuck, then insert more.
    
    Parameters
    ----------
    packer : WaterPacker
        The water packer instance.
    system : Any
        Input system (ASE Atoms, etc.).
    n_waters : int
        Target number of waters.
    relax_after_waters : int
        Start trying relaxation after this many waters placed.
    relax_every_failures : int
        Trigger relaxation after this many consecutive insertion failures.
    max_cycles : int
        Maximum insert-relax cycles.
    
    Returns
    -------
    result : Same type as input
        System with packed waters.
    """
    from .converters import detect_input_type, from_ase, to_ase
    from .spatial_hash import SpatialHashGrid
    from .volume_estimation import generate_candidate_batch
    from .water_geometry import get_water_template
    
    # Convert input
    input_type = detect_input_type(system)
    if input_type == 'ase':
        from .converters import from_ase
        data = from_ase(system)
    else:
        data = system.copy() if isinstance(system, dict) else from_ase(system)
    
    cell = np.array(data['cell'])
    substrate_positions = np.array(data['positions'])
    substrate_species = list(data['species'])
    
    # Initialize
    water_symbols, water_template = get_water_template()
    grid = SpatialHashGrid(cell, grid_spacing=3.0)
    grid.add_multiple(substrate_positions, substrate_species)
    
    relaxer = WaterRelaxer(cell, packer.distance_manager)
    
    placed_waters = []  # List of (3, 3) water coords
    placed_species = []  # List of ['O', 'H', 'H']
    
    cycle = 0
    consecutive_failures = 0
    
    while len(placed_waters) < n_waters and cycle < max_cycles:
        # Phase 1: Try batch insertion
        batch_placed = 0
        
        for _ in range(5):  # 5 batch attempts per cycle
            if len(placed_waters) >= n_waters:
                break
            
            candidates, rotations = generate_candidate_batch(
                cell, substrate_positions, substrate_species,
                packer.distance_manager,
                batch_size=packer.batch_size,
            )
            
            for center_pos, rotation in zip(candidates, rotations):
                if len(placed_waters) >= n_waters:
                    break
                
                water_coords = water_template @ rotation.T + center_pos
                
                # Check collisions
                has_collision = False
                for atom_pos, atom_sp in zip(water_coords, water_symbols):
                    if grid.check_collision(
                        atom_pos, atom_sp,
                        packer.distance_manager.get_min_distance
                    ):
                        has_collision = True
                        break
                
                if not has_collision:
                    placed_waters.append(water_coords)
                    placed_species.append(water_symbols.copy())
                    for atom_pos, atom_sp in zip(water_coords, water_symbols):
                        grid.add(atom_pos, atom_sp)
                    batch_placed += 1
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
        
        # Phase 2: Relax if stuck
        if (batch_placed == 0 or consecutive_failures > relax_every_failures) \
           and len(placed_waters) >= relax_after_waters:
            
            # Rebuild grid after relaxation
            grid.clear()
            grid.add_multiple(substrate_positions, substrate_species)
            
            water_array = np.array(placed_waters) if placed_waters else np.zeros((0, 3, 3))
            relaxed, n_accepted = relaxer.relax_waters(
                water_array,
                placed_species,
                substrate_positions,
                substrate_species,
            )
            
            if n_accepted > 0:
                placed_waters = list(relaxed)
                # Re-add relaxed waters to grid
                for water_coords, species_list in zip(placed_waters, placed_species):
                    for pos, sp in zip(water_coords, species_list):
                        grid.add(pos, sp)
                
                consecutive_failures = 0  # Reset after relaxation
        
        cycle += 1
    
    # Build result
    all_positions = substrate_positions
    all_species = substrate_species.copy()
    
    if placed_waters:
        water_flat = np.vstack(placed_waters)
        all_positions = np.vstack([substrate_positions, water_flat])
        for sp_list in placed_species:
            all_species.extend(sp_list)
    
    result_data = {
        'cell': cell,
        'positions': all_positions,
        'species': all_species,
        'pbc': data.get('pbc', True),
    }
    
    if input_type == 'ase':
        from .converters import to_ase
        return to_ase(result_data)
    return result_data
