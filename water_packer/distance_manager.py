"""Distance constraint management for water packing."""

from typing import Dict, Optional, Tuple


class DistanceManager:
    """
    Manage species-specific minimum distances.
    
    Default distances are based on typical vdW radii sums, with values
    appropriate for MD simulations.
    """
    
    # Default minimum distances based on vdW radii sums
    # These are contact distances, not equilibrium distances
    DEFAULT_DISTANCES: Dict[Tuple[str, str], float] = {
        # Water-water (inter-molecular)
        ('O', 'O'): 2.4,    # O-O water-water relaxed to 2.4A per user request
        ('O', 'H'): 1.6,    # O-H non-bonded
        ('H', 'H'): 1.5,    # H-H non-bonded
        
        # Water with common substrate atoms
        ('O', 'Mg'): 2.0,   # O-Mg (updated for realistic solvation)
        ('H', 'Mg'): 2.0,   # H-Mg
        ('O', 'Ca'): 2.4,   # O-Ca
        ('H', 'Ca'): 2.2,   # H-Ca
        ('O', 'Si'): 2.6,   # O-Si (silica)
        ('H', 'Si'): 2.2,   # H-Si
        ('O', 'Al'): 2.4,   # O-Al
        ('H', 'Al'): 2.0,   # H-Al
        ('O', 'Fe'): 2.3,   # O-Fe
        ('H', 'Fe'): 2.0,   # H-Fe
        ('O', 'Na'): 2.3,   # O-Na
        ('H', 'Na'): 2.0,   # H-Na
        ('O', 'K'): 2.6,    # O-K
        ('H', 'K'): 2.2,    # H-K
        ('O', 'C'): 2.6,    # O-C
        ('H', 'C'): 2.2,    # H-C
        ('O', 'N'): 2.6,    # O-N
        ('H', 'N'): 2.0,    # H-N
    }
    
    def __init__(
        self,
        default_distance: float = 2.0,
        pairwise_distances: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        """
        Initialize distance manager.
        
        Parameters
        ----------
        default_distance : float
            Fallback minimum distance for pairs not in lookup tables.
        pairwise_distances : dict, optional
            User-specified distances that override defaults.
            Keys are (species1, species2) tuples (order doesn't matter).
        """
        self.default_distance = default_distance
        self.user_distances = pairwise_distances or {}
    
    def _normalize_pair(self, species1: str, species2: str) -> Tuple[str, str]:
        """Normalize species pair to canonical order (alphabetical)."""
        return tuple(sorted([species1, species2]))
    
    def get_min_distance(self, species1: str, species2: str) -> float:
        """
        Get minimum distance for a species pair.
        
        Priority: user-specified > defaults > fallback
        
        Parameters
        ----------
        species1, species2 : str
            Element symbols.
        
        Returns
        -------
        float
            Minimum allowed distance in Angstrom.
        """
        pair = self._normalize_pair(species1, species2)
        
        # Check user overrides first
        if pair in self.user_distances:
            return self.user_distances[pair]
        # Also check reverse order in user distances
        reverse_pair = (pair[1], pair[0])
        if reverse_pair in self.user_distances:
            return self.user_distances[reverse_pair]
        
        # Check defaults
        if pair in self.DEFAULT_DISTANCES:
            return self.DEFAULT_DISTANCES[pair]
        if reverse_pair in self.DEFAULT_DISTANCES:
            return self.DEFAULT_DISTANCES[reverse_pair]
        
        # Fallback
        return self.default_distance
    
    def get_exclusion_radius(self, species: str) -> float:
        """
        Get the half-distance exclusion radius for substrate atom.
        
        This is half the typical water-O to substrate distance,
        representing the substrate's "share" of the interface exclusion.
        
        Parameters
        ----------
        species : str
            Element symbol of substrate atom.
        
        Returns
        -------
        float
            Exclusion radius in Angstrom.
        """
        # Use O-species distance as reference (O is the water center)
        full_distance = self.get_min_distance('O', species)
        return full_distance / 2.0
