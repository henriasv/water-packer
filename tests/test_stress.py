"""Heavy stress tests for water packer performance."""

import time
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from water_packer import WaterPacker


@pytest.mark.slow
def test_large_water_box():
    """Test packing 1000+ waters in a large box."""
    # 30×30×30 Å box → ~27000 Å³ → ~900 waters at 1 g/cm³
    box = Atoms(cell=[30, 30, 30], pbc=True)
    
    packer = WaterPacker(water_density=1.0, seed=42)
    
    start = time.time()
    result = packer.pack(box)
    elapsed = time.time() - start
    
    n_waters = len(result) // 3
    print(f"\n  Placed {n_waters} waters in {elapsed:.2f}s ({n_waters/elapsed:.1f} waters/s)")
    
    assert n_waters > 500  # Should place most of them
    assert elapsed < 60.0  # Should complete in reasonable time


@pytest.mark.slow
def test_large_substrate():
    """Test packing around a large substrate (100+ atoms)."""
    # Create a large MgO slab
    mgo = bulk('MgO', 'rocksalt', a=4.2).repeat((3, 3, 3))
    n_substrate = len(mgo)
    
    # Expand cell for water
    cell = mgo.get_cell()
    cell[2, 2] += 15.0
    mgo.set_cell(cell)
    mgo.center(axis=2)
    
    packer = WaterPacker(
        pairwise_distances={('O', 'Mg'): 2.0},
        water_density=1.0,
        seed=42,
    )
    
    start = time.time()
    result = packer.pack(mgo)
    elapsed = time.time() - start
    
    n_waters = (len(result) - n_substrate) // 3
    print(f"\n  Substrate: {n_substrate} atoms")
    print(f"  Placed {n_waters} waters in {elapsed:.2f}s")
    
    assert n_waters > 20
    assert len(result) == n_substrate + n_waters * 3


@pytest.mark.slow
def test_high_density_packing():
    """Test packing at high density with tight constraints."""
    box = Atoms(cell=[20, 20, 20], pbc=True)
    
    packer = WaterPacker(
        min_distance=2.2,  # Tighter than default
        water_density=1.0,
        seed=42,
    )
    
    start = time.time()
    result = packer.pack(box)
    elapsed = time.time() - start
    
    n_waters = len(result) // 3
    print(f"\n  High density: {n_waters} waters in {elapsed:.2f}s")
    
    # Verify minimum distances are respected
    positions = result.get_positions()
    symbols = result.get_chemical_symbols()
    cell = result.get_cell()
    
    # Check a sample of pairs
    n_checks = min(1000, len(positions) * (len(positions) - 1) // 2)
    pairs = np.random.choice(len(positions), size=(n_checks, 2), replace=False)
    
    for i, j in pairs:
        if i // 3 == j // 3:  # Skip same molecule
            continue
        
        diff = positions[i] - positions[j]
        frac = np.linalg.solve(cell.T, diff)
        frac -= np.round(frac)
        diff = np.dot(frac, cell)
        dist = np.linalg.norm(diff)
        
        # Should respect min distance (with small tolerance)
        assert dist >= 1.3, f"Atoms too close: {dist:.2f} Å"


@pytest.mark.slow
def test_batch_efficiency():
    """Test that batch generation is efficient."""
    box = Atoms(cell=[25, 25, 25], pbc=True)
    
    packer = WaterPacker(
        water_density=1.0,
        seed=42,
        batch_size=50000,  # Large batch
    )
    
    start = time.time()
    result = packer.pack(box)
    elapsed = time.time() - start
    
    n_waters = len(result) // 3
    print(f"\n  Batch packing: {n_waters} waters in {elapsed:.2f}s")
    print(f"  Throughput: {n_waters/elapsed:.1f} waters/s")
    
    # Should place most expected waters
    expected = int(25**3 * 0.0334)
    assert n_waters > 0.7 * expected


@pytest.mark.slow
def test_monte_carlo_volume_accuracy():
    """Test Monte Carlo volume estimation accuracy."""
    from water_packer.volume_estimation import estimate_available_volume_mc
    from water_packer.distance_manager import DistanceManager
    
    # Empty box - should be ~full volume
    box = Atoms(cell=[10, 10, 10], pbc=True)
    dm = DistanceManager()
    
    estimated = estimate_available_volume_mc(
        np.zeros((0, 3)), [], box.get_cell(), dm, n_probes=100000
    )
    actual = 1000.0
    
    print(f"\n  Empty box: estimated={estimated:.1f} Å³, actual={actual:.1f} Å³")
    assert abs(estimated - actual) / actual < 0.02  # Within 2%
    
    # Box with substrate
    substrate = Atoms('Mg', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    estimated_sub = estimate_available_volume_mc(
        substrate.get_positions(),
        substrate.get_chemical_symbols(),
        substrate.get_cell(),
        dm,
        n_probes=100000,
    )
    
    print(f"  With substrate: estimated={estimated_sub:.1f} Å³ (reduced)")
    assert estimated_sub < estimated  # Should be reduced


@pytest.mark.slow
def test_reproducibility_large():
    """Test that large systems are reproducible with same seed."""
    box = Atoms(cell=[20, 20, 20], pbc=True)
    
    packer1 = WaterPacker(seed=123)
    result1 = packer1.pack(box, n_waters=100)
    
    packer2 = WaterPacker(seed=123)
    result2 = packer2.pack(box, n_waters=100)
    
    assert np.allclose(result1.get_positions(), result2.get_positions())


@pytest.mark.slow
def test_non_orthorhombic_large():
    """Test with non-orthorhombic cell."""
    cell = np.array([
        [15.0, 0.0, 0.0],
        [3.0, 15.0, 0.0],
        [2.0, 2.0, 15.0],
    ])
    
    box = Atoms(cell=cell, pbc=True)
    packer = WaterPacker(seed=42)
    
    start = time.time()
    result = packer.pack(box, n_waters=200)
    elapsed = time.time() - start
    
    n_placed = len(result) // 3
    print(f"\n  Non-orthorhombic: {n_placed} waters in {elapsed:.2f}s")
    
    assert n_placed >= 150


def test_performance_comparison():
    """Compare performance metrics."""
    box = Atoms(cell=[15, 15, 15], pbc=True)
    
    packer = WaterPacker(seed=42)
    
    start = time.time()
    result = packer.pack(box)
    elapsed = time.time() - start
    
    n_waters = len(result) // 3
    throughput = n_waters / elapsed
    
    print(f"\n  Performance Summary:")
    print(f"  - Waters placed: {n_waters}")
    print(f"  - Time: {elapsed:.3f}s")
    print(f"  - Throughput: {throughput:.1f} waters/s")
    
    # Should be reasonably fast
    assert throughput > 10  # At least 10 waters/second


if __name__ == "__main__":
    # Run heavy tests manually
    print("Running heavy stress tests...")
    print("\n" + "="*60)
    test_large_water_box()
    print("="*60)
    test_large_substrate()
    print("="*60)
    test_high_density_packing()
    print("="*60)
    test_batch_efficiency()
    print("="*60)
    test_monte_carlo_volume_accuracy()
    print("="*60)
    test_performance_comparison()
    print("="*60)
    print("\nAll stress tests completed!")
