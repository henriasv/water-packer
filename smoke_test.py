"""Quick smoke test for optimized packer."""

from water_packer import WaterPacker
from ase import Atoms
import time

print("Testing optimized water packer...")
print("="*60)

# Test 1: Small box
print("\n1. Small box (10x10x10 Å)...")
box = Atoms(cell=[10, 10, 10], pbc=True)
packer = WaterPacker(seed=42)

start = time.time()
result = packer.pack(box, n_waters=20)
elapsed = time.time() - start

print(f"   ✓ Placed {len(result)//3} waters in {elapsed:.3f}s")

# Test 2: Medium box
print("\n2. Medium box (20x20x20 Å)...")
box = Atoms(cell=[20, 20, 20], pbc=True)
packer = WaterPacker(seed=42)

start = time.time()
result = packer.pack(box)
elapsed = time.time() - start

n_waters = len(result) // 3
print(f"   ✓ Placed {n_waters} waters in {elapsed:.3f}s")
print(f"   Throughput: {n_waters/elapsed:.1f} waters/s")

# Test 3: With substrate
print("\n3. With MgO substrate...")
from ase.build import bulk
mgo = bulk('MgO', 'rocksalt', a=4.2)
cell = mgo.get_cell()
cell[2, 2] += 10.0
mgo.set_cell(cell)
mgo.center(axis=2)

packer = WaterPacker(pairwise_distances={('O', 'Mg'): 2.0}, seed=42)
start = time.time()
result = packer.pack(mgo, n_waters=10)
elapsed = time.time() - start

print(f"   ✓ Placed {(len(result)-len(mgo))//3} waters in {elapsed:.3f}s")

print("\n" + "="*60)
print("All smoke tests passed!")
