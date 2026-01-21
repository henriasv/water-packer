"""Tests for WaterPacker core functionality."""

import numpy as np
import pytest
from ase import Atoms

from water_packer import WaterPacker
from water_packer.constants import WATER_DENSITY_FACTOR


def test_water_packer_init():
    """Test WaterPacker initialization."""
    packer = WaterPacker(min_distance=2.0, water_density=1.0, seed=42)
    
    assert packer.water_density == 1.0
    assert packer.distance_manager.default_distance == 2.0


def test_empty_box_packing():
    """Test packing water in an empty box."""
    # 10 Å cube
    atoms = Atoms(cell=[10, 10, 10], pbc=True)
    
    packer = WaterPacker(water_density=1.0, seed=42)
    result = packer.pack(atoms, n_waters=10)
    
    # Should have 30 atoms (10 waters × 3 atoms each)
    assert len(result) == 30
    
    # Check that we have O and H atoms
    symbols = result.get_chemical_symbols()
    assert symbols.count('O') == 10
    assert symbols.count('H') == 20


def test_density_calculation():
    """Test automatic density-based water count calculation."""
    # 10×10×10 Å box = 1000 Å³
    # At 1 g/cm³: 1000 * 0.0334 ≈ 33 waters
    atoms = Atoms(cell=[10, 10, 10], pbc=True)
    
    packer = WaterPacker(water_density=1.0, seed=42)
    result = packer.pack(atoms)  # n_waters=None
    
    # Should be roughly 33 waters (depends on packing success)
    n_waters = (len(result)) // 3
    assert 25 < n_waters < 40, f"Expected ~33 waters, got {n_waters}"


def test_minimum_distance_enforcement():
    """Test that minimum distances are respected between different molecules."""
    atoms = Atoms(cell=[10, 10, 10], pbc=True)
    
    packer = WaterPacker(min_distance=2.0, seed=42)
    result = packer.pack(atoms, n_waters=5)
    
    positions = result.get_positions()
    symbols = result.get_chemical_symbols()
    cell = result.get_cell()
    
    # Check inter-molecular distances (skip bonded atoms within same water)
    # Water molecules are 3 atoms each: [O, H, H], [O, H, H], ...
    n_molecules = len(positions) // 3
    
    for mol1 in range(n_molecules):
        for mol2 in range(mol1 + 1, n_molecules):
            # Get atoms from each molecule
            mol1_indices = range(mol1 * 3, mol1 * 3 + 3)
            mol2_indices = range(mol2 * 3, mol2 * 3 + 3)
            
            # Check all atom pairs between the two molecules
            for i in mol1_indices:
                for j in mol2_indices:
                    diff = positions[i] - positions[j]
                    # Apply minimum image
                    frac = np.linalg.solve(cell.T, diff)
                    frac -= np.round(frac)
                    diff = np.dot(frac, cell)
                    dist = np.linalg.norm(diff)
                    
                    # Allow some tolerance, but should respect minimum distances
                    # O-O should be > 2.6 Å, H-H > 1.5 Å, O-H > 1.6 Å
                    s1, s2 = symbols[i], symbols[j]
                    expected_min = {
                        ('O', 'O'): 2.4,  # Slightly lower tolerance
                        ('H', 'H'): 1.3,
                        ('O', 'H'): 1.4,
                        ('H', 'O'): 1.4,
                    }.get((s1, s2), 1.3)
                    
                    assert dist >= expected_min, \
                        f"Molecules {mol1} and {mol2}: {s1}-{s2} too close: {dist:.2f} Å"



def test_pairwise_distances():
    """Test species-specific pairwise distances."""
    # Create a Mg atom
    atoms = Atoms('Mg', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    
    # Set large O-Mg distance
    packer = WaterPacker(
        min_distance=2.0,
        pairwise_distances={('O', 'Mg'): 3.5},
        seed=42,
    )
    
    result = packer.pack(atoms, n_waters=3)
    
    # Check that water O atoms are at least 3.5 Å from Mg
    mg_pos = atoms.get_positions()[0]
    cell = result.get_cell()
    
    for symbol, pos in zip(result.get_chemical_symbols(), result.get_positions()):
        if symbol == 'O':
            diff = pos - mg_pos
            # Apply minimum image
            frac = np.linalg.solve(cell.T, diff)
            frac -= np.round(frac)
            diff = np.dot(frac, cell)
            dist = np.linalg.norm(diff)
            
            assert dist >= 3.4, f"Water O too close to Mg: {dist:.2f} Å"


def test_reproducibility():
    """Test that packing is reproducible with same seed."""
    atoms = Atoms(cell=[10, 10, 10], pbc=True)
    
    packer1 = WaterPacker(seed=42)
    result1 = packer1.pack(atoms, n_waters=5)
    
    packer2 = WaterPacker(seed=42)
    result2 = packer2.pack(atoms, n_waters=5)
    
    # Should get identical positions
    assert np.allclose(result1.get_positions(), result2.get_positions())


def test_excluded_volume_calculation():
    """Test that substrate atoms reduce available volume."""
    # Empty box
    empty = Atoms(cell=[10, 10, 10], pbc=True)
    packer_empty = WaterPacker(water_density=1.0, seed=42)
    result_empty = packer_empty.pack(empty)
    n_waters_empty = len(result_empty) // 3
    
    # Box with large substrate atom
    substrate = Atoms('Mg', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    packer_sub = WaterPacker(water_density=1.0, seed=42)
    result_sub = packer_sub.pack(substrate)
    n_waters_sub = (len(result_sub) - 1) // 3  # Subtract the Mg
    
    # Should have fewer waters with substrate
    assert n_waters_sub < n_waters_empty


def test_output_type_preservation():
    """Test that output type matches input type."""
    from ase import Atoms
    
    atoms = Atoms(cell=[10, 10, 10], pbc=True)
    packer = WaterPacker(seed=42)
    
    result = packer.pack(atoms, n_waters=5)
    assert isinstance(result, Atoms)
    
    # Test with dict input
    data = {'cell': np.diag([10, 10, 10]), 'positions': np.zeros((0, 3)), 'species': []}
    result_dict = packer.pack(data, n_waters=5)
    assert isinstance(result_dict, dict)
