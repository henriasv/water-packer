"""Monte Carlo volume estimation for water packing."""

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional


def estimate_available_volume_mc(
    positions: np.ndarray,
    species: List[str],
    cell: np.ndarray,
    distance_manager,
    n_probes: int = 100000,
    water_radius: float = 1.3,  # Half of O-O distance (2.6 Å)
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Estimate available volume using Monte Carlo sampling.
    
    Avoids overlap issues from analytical sphere subtraction by randomly
    sampling points in the cell and checking if they're accessible.
    
    Parameters
    ----------
    positions : np.ndarray
        (N, 3) substrate atom positions.
    species : list of str
        Element symbols for each position.
    cell : np.ndarray
        (3, 3) cell matrix.
    distance_manager : DistanceManager
        For getting exclusion radii.
    n_probes : int
        Number of Monte Carlo probe points.
    water_radius : float
        Effective radius of water oxygen (= O-O distance / 2).
    
    Returns
    -------
    float
        Estimated available volume in Å³.
    """
    if len(positions) == 0:
        # Empty cell - all volume available
        return np.abs(np.linalg.det(cell))
    
    # Use provided rng or create default
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate random probe points in fractional coordinates
    frac_coords = rng.random((n_probes, 3))
    cart_coords = np.dot(frac_coords, cell)
    
    # Build KDTree with periodic boundaries
    # For orthorhombic cells, cKDTree can use boxsize
    cell_diagonal = np.diag(cell)
    is_orthorhombic = np.allclose(cell, np.diag(cell_diagonal))
    
    if is_orthorhombic:
        tree = cKDTree(positions, boxsize=cell_diagonal)
        dists, indices = tree.query(cart_coords)
    else:
        # For non-orthorhombic, manually apply MIC
        tree = cKDTree(positions)
        dists, indices = tree.query(cart_coords)
        # Note: This is approximate for non-orthorhombic cells
        # A full MIC would require wrapping each probe point
    
    # Check if each probe is "free"
    # A probe is free if distance to nearest substrate > R_substrate + R_probe
    free_count = 0
    for i in range(n_probes):
        nearest_species = species[indices[i]]
        exclusion_radius = distance_manager.get_exclusion_radius(nearest_species)
        min_required_dist = exclusion_radius + water_radius
        
        if dists[i] > min_required_dist:
            free_count += 1
    
    # Estimate available volume
    cell_volume = np.abs(np.linalg.det(cell))
    available_volume = cell_volume * free_count / n_probes
    
    return available_volume


def generate_candidate_batch(
    cell: np.ndarray,
    substrate_positions: np.ndarray,
    substrate_species: List[str],
    distance_manager,
    batch_size: int = 50000,
    region: dict = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a large batch of candidate positions and cull against substrate.
    
    Uses cKDTree for vectorized distance checking - much faster than looping.
    
    Parameters
    ----------
    cell : np.ndarray
        (3, 3) cell matrix.
    substrate_positions : np.ndarray
        (N, 3) substrate atom positions.
    substrate_species : list of str
        Element symbols.
    distance_manager : DistanceManager
        For getting minimum distances.
    batch_size : int
        Number of candidates to generate.
    region : dict, optional
        Spatial constraints like {'z_min': 5.0, 'z_max': 15.0}.
    
    Returns
    -------
    candidates : np.ndarray
        (M, 3) array of safe candidate O positions.
    rotations : np.ndarray
        (M, 3, 3) array of rotation matrices for each candidate.
    """
    from scipy.spatial.transform import Rotation
    
    # Use provided rng or create default
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate random fractional coordinates
    frac_coords = rng.random((batch_size, 3))
    
    # Apply region constraints if specified
    if region is not None:
        inv_cell = np.linalg.inv(cell)
        if 'z_min' in region and 'z_max' in region:
            # Convert z bounds to fractional
            z_min_frac = np.dot(inv_cell, [0, 0, region['z_min']])[2]
            z_max_frac = np.dot(inv_cell, [0, 0, region['z_max']])[2]
            frac_coords[:, 2] = z_min_frac + frac_coords[:, 2] * (z_max_frac - z_min_frac)
    
    # Convert to Cartesian
    cart_coords = np.dot(frac_coords, cell)
    
    # Generate random rotations for each candidate
    rotations = Rotation.random(batch_size, random_state=rng).as_matrix()
    
    if len(substrate_positions) == 0:
        # No substrate - all candidates are safe
        return cart_coords, rotations
    
    # Build KDTree for substrate
    cell_diagonal = np.diag(cell)
    is_orthorhombic = np.allclose(cell, np.diag(cell_diagonal))
    
    if is_orthorhombic:
        tree = cKDTree(substrate_positions, boxsize=cell_diagonal)
    else:
        tree = cKDTree(substrate_positions)
    
    # Query all candidates at once (vectorized!)
    dists, indices = tree.query(cart_coords)
    
    # Determine minimum required distance for each candidate
    # This is the distance from water-O to the nearest substrate atom
    min_required_dists = np.array([
        distance_manager.get_min_distance('O', substrate_species[idx])
        for idx in indices
    ])
    
    # Keep only candidates that are far enough from substrate
    safe_mask = dists >= min_required_dists
    safe_candidates = cart_coords[safe_mask]
    safe_rotations = rotations[safe_mask]
    
    return safe_candidates, safe_rotations
