"""System type converters for ASE, hyobj, and raw arrays."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Type alias for internal representation
SystemDict = Dict[str, Any]


def from_ase(atoms) -> SystemDict:
    """
    Convert ASE Atoms to internal dict format.
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object.
    
    Returns
    -------
    dict
        Internal representation with 'cell', 'positions', 'species'.
    """
    cell = np.array(atoms.get_cell())
    positions = atoms.get_positions()
    species = atoms.get_chemical_symbols()
    
    return {
        'cell': cell,
        'positions': positions,
        'species': species,
        'pbc': atoms.get_pbc(),
    }


def to_ase(data: SystemDict):
    """
    Convert internal dict to ASE Atoms.
    
    Parameters
    ----------
    data : dict
        Internal representation.
    
    Returns
    -------
    ase.Atoms
        ASE Atoms object.
    """
    from ase import Atoms
    
    pbc = data.get('pbc', True)
    atoms = Atoms(
        symbols=data['species'],
        positions=data['positions'],
        cell=data['cell'],
        pbc=pbc,
    )
    return atoms


def from_hyobj(system) -> SystemDict:
    """
    Convert hyobj PeriodicSystem to internal dict format.
    
    Parameters
    ----------
    system : hyobj.PeriodicSystem
        Hylleraas PeriodicSystem object.
    
    Returns
    -------
    dict
        Internal representation.
    """
    # hyobj uses .coordinates for positions and .pbc for cell
    # Units should already be in Angstrom if using 'metal' units
    cell = np.array(system.pbc)
    positions = np.array(system.coordinates)
    species = list(system.atoms)
    
    return {
        'cell': cell,
        'positions': positions,
        'species': species,
        'pbc': True,  # PeriodicSystem is always periodic
        '_hyobj_units': getattr(system, 'units', None),
    }


def to_hyobj(data: SystemDict, units: str = 'metal'):
    """
    Convert internal dict to hyobj PeriodicSystem.
    
    Parameters
    ----------
    data : dict
        Internal representation.
    units : str
        Unit system for hyobj (default 'metal' = Angstrom/eV).
    
    Returns
    -------
    hyobj.PeriodicSystem
        Hylleraas PeriodicSystem object.
    """
    try:
        from hyobj import PeriodicSystem, Units
    except ImportError:
        raise ImportError(
            "hyobj is required for PeriodicSystem output. "
            "Install with: pip install water_packer[hyobj]"
        )
    
    mol_input = {
        'atoms': data['species'],
        'coordinates': data['positions'],
    }
    
    return PeriodicSystem(mol_input, pbc=data['cell'], units=Units(units))


def from_arrays(
    cell: np.ndarray,
    positions: np.ndarray,
    species: Optional[List[str]] = None,
) -> SystemDict:
    """
    Convert raw arrays to internal dict format.
    
    Parameters
    ----------
    cell : np.ndarray
        (3, 3) cell vectors or (3,) for orthorhombic.
    positions : np.ndarray
        (N, 3) atomic positions in Angstrom.
    species : list of str, optional
        Element symbols. If None, all atoms are 'X'.
    
    Returns
    -------
    dict
        Internal representation.
    """
    cell = np.asarray(cell)
    if cell.shape == (3,):
        cell = np.diag(cell)
    
    positions = np.asarray(positions)
    n_atoms = len(positions)
    
    if species is None:
        species = ['X'] * n_atoms
    
    return {
        'cell': cell,
        'positions': positions,
        'species': list(species),
        'pbc': True,
    }


def to_arrays(data: SystemDict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert internal dict to raw arrays.
    
    Parameters
    ----------
    data : dict
        Internal representation.
    
    Returns
    -------
    cell : np.ndarray
        (3, 3) cell vectors.
    positions : np.ndarray
        (N, 3) atomic positions.
    species : list of str
        Element symbols.
    """
    return data['cell'], data['positions'], data['species']


def detect_input_type(system) -> str:
    """
    Detect the type of input system.
    
    Returns
    -------
    str
        One of 'ase', 'hyobj', 'dict', 'unknown'.
    """
    # Check for ASE Atoms
    try:
        from ase import Atoms
        if isinstance(system, Atoms):
            return 'ase'
    except ImportError:
        pass
    
    # Check for hyobj PeriodicSystem
    try:
        from hyobj import PeriodicSystem
        if isinstance(system, PeriodicSystem):
            return 'hyobj'
    except ImportError:
        pass
    
    # Check for dict format
    if isinstance(system, dict):
        if 'positions' in system or 'coordinates' in system:
            return 'dict'
    
    return 'unknown'
