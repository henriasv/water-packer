"""Integration tests with real structures."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from water_packer import WaterPacker
try:
    import hyobj
    HAS_HYOBJ = True
except ImportError:
    HAS_HYOBJ = False

def test_pack_around_mgo_slab():
    """Test packing water around a MgO slab."""
    # Create a MgO slab (approx 30 atoms) per user request
    # Primitive cell has 2 atoms. 4x4x1 repeat = 32 atoms.
    mgo = bulk('MgO', 'rocksalt', a=4.2).repeat((4, 4, 1))
    
    # Expand in z to add vacuum
    cell = mgo.get_cell()
    print(f"  Slab atoms: {len(mgo)}")
    cell[2, 2] += 12.0  # Add 12 Å vacuum for water gap
    mgo.set_cell(cell)
    mgo.center(axis=2)
    
    # Pack water
    packer = WaterPacker(
        min_distance=2.0,
        pairwise_distances={('O', 'Mg'): 2.2, ('O', 'O'): 2.6},
        water_density=1.0,
        seed=42,
    )
    
    # Pack water based on density=1.0
    result = packer.pack(mgo, n_waters=None)
    
    # Should have original atoms plus water
    n_substrate = len(mgo)
    n_total = len(result)
    n_added = n_total - n_substrate
    
    assert n_added > 0
    assert n_added % 3 == 0  # Should be multiples of 3 (water molecules)


def test_water_box():
    """Test creating a pure water box."""
    atoms = Atoms(cell=[15, 15, 15], pbc=True)
    
    packer = WaterPacker(water_density=1.0, seed=42)
    result = packer.pack(atoms)
    
    # Should have significant number of waters
    n_waters = len(result) // 3
    assert n_waters > 30
    
    # Check chemistry
    symbols = result.get_chemical_symbols()
    assert symbols.count('O') == n_waters
    assert symbols.count('H') == 2 * n_waters


def test_non_orthorhombic_cell():
    """Test packing with non-orthorhombic cell."""
    # Create a triclinic cell
    cell = np.array([
        [10.0, 0.0, 0.0],
        [2.0, 10.0, 0.0],
        [1.0, 1.0, 10.0],
    ])
    
    atoms = Atoms(cell=cell, pbc=True)
    
    packer = WaterPacker(seed=42)
    result = packer.pack(atoms, n_waters=10)
    
    assert len(result) == 30
    # All positions should be within the cell (roughly)
    positions = result.get_positions()
    assert np.all(positions >= -1)  # Small tolerance for numerical issues


def test_partial_packing():
    """Test when not all waters can be placed."""
    # Very small box with tight constraints
    atoms = Atoms(cell=[5, 5, 5], pbc=True)
    
    packer = WaterPacker(
        min_distance=3.0,  # Very tight constraints
        seed=42,
        batch_size=1000,   # Small batch for fast failure
    )
    
    # Request a few waters - won't all fit in 5x5x5 with 3.0Å min dist
    result = packer.pack(atoms, n_waters=20)
    
    n_placed = len(result) // 3
    # Should place some but not all
    assert 0 < n_placed < 20


def test_with_existing_structure():
    """Test that existing atoms are preserved and water is added."""
    from ase import Atoms
    # Single atom in a large box (much easier to pack)
    atoms = Atoms('Si', positions=[[5, 5, 5]], cell=[15, 15, 15], pbc=True)
    
    packer = WaterPacker(
        pairwise_distances={('O', 'Si'): 2.0},
        water_density=1.0,
        seed=42,
    )
    
    # Request 2 waters
    result = packer.pack(atoms, n_waters=2)
    
    # Should have 1 Si + 2 waters (7 atoms total)
    assert len(result) == 7
    assert result.get_chemical_symbols()[0] == 'Si'
    assert result.get_chemical_symbols().count('O') == 2
    
    # Verify Si atom is still at [5, 5, 5]
    assert np.allclose(result.get_positions()[0], [5, 5, 5])


@pytest.mark.skipif(not HAS_HYOBJ, reason="hyobj not installed")
def test_hyobj_input():
    """Test with hyobj PeriodicSystem input."""
    from hyobj import PeriodicSystem, Units
    
    mol_input = {
        'atoms': ['Mg', 'O'],
        'coordinates': [[0, 0, 0], [2, 2, 2]],
    }
    cell = np.diag([10, 10, 10])
    
    system = PeriodicSystem(mol_input, pbc=cell, units=Units('metal'))
    
    packer = WaterPacker(seed=42)
    result = packer.pack(system, n_waters=3)
    
    # Should return PeriodicSystem
    assert isinstance(result, PeriodicSystem)
    assert result.num_atoms > system.num_atoms
