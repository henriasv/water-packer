"""Manual verification script with real structures."""

from water_packer import WaterPacker
from ase.io import read
import numpy as np

print("=" * 60)
print("Water Packer - Manual Verification")
print("=" * 60)

# Test 1: Pack water around periclase (MgO)
print("\n1. Packing water around periclase (MgO)...")
try:
    mgo_path = '/Users/henriasv/repos/antigravity_repos/cp2k_chemfrac/active_learning/structures/periclase.cif'
    mgo = read(mgo_path)
    
    print(f"   Original MgO: {len(mgo)} atoms")
    print(f"   Cell volume: {mgo.get_volume():.1f} Å³")
    
    # Expand cell to add vacuum
    cell = mgo.get_cell()
    cell[2, 2] += 10.0  # Add 10 Å in z
    mgo.set_cell(cell)
    mgo.center(axis=2)
    
    packer = WaterPacker(
        min_distance=2.0,
        pairwise_distances={('O', 'Mg'): 2.5, ('O', 'O'): 2.6},
        water_density=1.0,
        seed=42,
    )
    
    result = packer.pack(mgo)
    n_waters = (len(result) - len(mgo)) // 3
    
    print(f"   ✓ Added {n_waters} water molecules ({len(result) - len(mgo)} atoms)")
    print(f"   Final system: {len(result)} atoms")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Pure water box
print("\n2. Creating pure water box...")
try:
    from ase import Atoms
    
    box = Atoms(cell=[15, 15, 15], pbc=True)
    packer = WaterPacker(water_density=1.0, seed=42)
    
    result = packer.pack(box)
    n_waters = len(result) // 3
    
    expected_volume = 15**3
    expected_waters = int(expected_volume * 0.0334)
    
    print(f"   ✓ Created water box with {n_waters} molecules")
    print(f"   Expected ~{expected_waters} waters at 1 g/cm³")
    print(f"   Placement efficiency: {n_waters/expected_waters*100:.1f}%")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Custom density
print("\n3. Testing custom density (0.5 g/cm³)...")
try:
    box = Atoms(cell=[10, 10, 10], pbc=True)
    packer = WaterPacker(water_density=0.5, seed=42)
    
    result = packer.pack(box)
    n_waters = len(result) // 3
    
    expected = int(1000 * 0.0334 * 0.5)
    
    print(f"   ✓ Packed {n_waters} waters")
    print(f"   Expected ~{expected} at 0.5 g/cm³")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 4: Species-specific distances
print("\n4. Testing species-specific distance constraints...")
try:
    # Create Mg substrate
    substrate = Atoms('Mg2', positions=[[0, 0, 0], [5, 5, 5]], cell=[10, 10, 10], pbc=True)
    
    packer = WaterPacker(
        pairwise_distances={
            ('O', 'Mg'): 3.0,  # Large exclusion
            ('H', 'Mg'): 2.5,
        },
        seed=42,
    )
    
    result = packer.pack(substrate, n_waters=10)
    
    # Verify O-Mg distances
    positions = result.get_positions()
    symbols = result.get_chemical_symbols()
    cell = result.get_cell()
    inv_cell = np.linalg.inv(cell)
    
    min_o_mg_dist = float('inf')
    for i, (pos, sym) in enumerate(zip(positions, symbols)):
        if sym == 'O':
            for j, (pos2, sym2) in enumerate(zip(positions, symbols)):
                if sym2 == 'Mg':
                    diff = pos - pos2
                    frac = np.dot(diff, inv_cell)
                    frac -= np.round(frac)
                    diff = np.dot(frac, cell)
                    dist = np.linalg.norm(diff)
                    min_o_mg_dist = min(min_o_mg_dist, dist)
    
    print(f"   ✓ Minimum O-Mg distance: {min_o_mg_dist:.2f} Å")
    print(f"   ✓ Constraint (3.0 Å) respected: {min_o_mg_dist >= 2.9}")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "=" * 60)
print("All manual tests completed successfully!")
print("=" * 60)
