"""Tests for converters module."""

import numpy as np
import pytest

from water_packer.converters import (
    detect_input_type,
    from_arrays,
    from_ase,
    to_arrays,
    to_ase,
)
try:
    import hyobj
    HAS_HYOBJ = True
except ImportError:
    HAS_HYOBJ = False

def test_from_arrays():
    """Test conversion from raw arrays."""
    cell = np.array([10.0, 10.0, 10.0])
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    species = ['Mg', 'O']
    
    data = from_arrays(cell, positions, species)
    
    assert data['cell'].shape == (3, 3)
    assert np.allclose(data['cell'], np.diag([10.0, 10.0, 10.0]))
    assert data['positions'].shape == (2, 3)
    assert data['species'] == ['Mg', 'O']


def test_from_arrays_3x3_cell():
    """Test with 3x3 cell matrix."""
    cell = np.eye(3) * 10.0
    positions = np.array([[1.0, 1.0, 1.0]])
    species = ['H']
    
    data = from_arrays(cell, positions, species)
    
    assert np.allclose(data['cell'], cell)


def test_ase_roundtrip():
    """Test ASE Atoms round-trip conversion."""
    from ase import Atoms
    
    atoms = Atoms(
        symbols=['Mg', 'O'],
        positions=[[0, 0, 0], [2, 2, 2]],
        cell=[10, 10, 10],
        pbc=True,
    )
    
    # Convert to internal
    data = from_ase(atoms)
    
    # Convert back
    result = to_ase(data)
    
    assert len(result) == len(atoms)
    assert result.get_chemical_symbols() == atoms.get_chemical_symbols()
    assert np.allclose(result.get_positions(), atoms.get_positions())
    assert np.allclose(result.get_cell(), atoms.get_cell())


def test_arrays_roundtrip():
    """Test raw arrays round-trip."""
    cell = np.diag([10.0, 11.0, 12.0])
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    species = ['H', 'O']
    
    data = from_arrays(cell, positions, species)
    cell_out, pos_out, sp_out = to_arrays(data)
    
    assert np.allclose(cell_out, cell)
    assert np.allclose(pos_out, positions)
    assert sp_out == species


def test_detect_type_ase():
    """Test detection of ASE Atoms."""
    from ase import Atoms
    
    atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    
    assert detect_input_type(atoms) == 'ase'


def test_detect_type_dict():
    """Test detection of dict format."""
    data = {
        'positions': np.zeros((3, 3)),
        'cell': np.eye(3),
        'species': ['H', 'H', 'O'],
    }
    
    assert detect_input_type(data) == 'dict'


@pytest.mark.skipif(not HAS_HYOBJ, reason="hyobj not installed")
def test_hyobj_roundtrip():
    """Test hyobj PeriodicSystem round-trip (if hyobj available)."""
    from hyobj import PeriodicSystem, Units
    
    mol_input = {
        'atoms': ['Mg', 'O'],
        'coordinates': [[0, 0, 0], [2, 2, 2]],
    }
    cell = np.diag([10, 10, 10])
    
    system = PeriodicSystem(mol_input, pbc=cell, units=Units('metal'))
    
    # Convert to internal
    from water_packer.converters import from_hyobj, to_hyobj
    data = from_hyobj(system)
    
    # Convert back
    result = to_hyobj(data, units='metal')
    
    assert result.num_atoms == system.num_atoms
    assert np.allclose(result.coordinates, system.coordinates)
