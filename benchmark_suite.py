"""
Benchmark Suite for Water Packer.

Runs systematic tests across:
1. Densities: 0.8 to 1.1 g/cm³
2. Sizes: 10, 20, 30 Å (approximate characteristic length)
3. Systems: Pure Water, Periclase (MgO), Brucite (Mg(OH)2)

Verifies Brucite OH termination.
Exports results to CSV and structures to .xyz in benchmark_results/
"""

import time
import csv
import os
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write
from water_packer import WaterPacker, pack_with_relaxation
from water_packer.volume_estimation import estimate_available_volume_mc
from water_packer.constants import WATER_DENSITY_FACTOR

def verify_brucite_termination(atoms):
    """Check if the slab is terminated by H atoms on both Z-sides."""
    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())
    z_coords = positions[:, 2]
    
    min_z_idx = np.argmin(z_coords)
    max_z_idx = np.argmax(z_coords)
    
    top_symbol = symbols[max_z_idx]
    bottom_symbol = symbols[min_z_idx]
    
    print(f"      Surface check: Top={top_symbol} (z={z_coords[max_z_idx]:.2f}), Bottom={bottom_symbol} (z={z_coords[min_z_idx]:.2f})")
    
    # Ideally both should be H
    if top_symbol == 'H' and bottom_symbol == 'H':
        return True
    
    # If not, we might need to shift the cell or cut differently
    # But for now just warn
    if top_symbol != 'H' or bottom_symbol != 'H':
        print("      WARNING: Brucite slab might not be H-terminated!")
        return False
    return True

def make_pure_water(size):
    """Create cubic box of size L."""
    return Atoms(cell=[size, size, size], pbc=True)

def make_periclase(size):
    """Create MgO slab with width ~size."""
    # Unit cell constant a=4.21
    # n * 4.21 ~ size
    n = max(2, int(round(size / 4.21)))
    
    # Make cubic slab
    slab = bulk('MgO', 'rocksalt', a=4.21, cubic=True).repeat((n, n, 2))
    
    # Add vacuum
    cell = slab.get_cell()
    cell[2, 2] += max(10.0, size * 0.5) # Scale vacuum with size slightly, min 10
    slab.set_cell(cell)
    slab.center(axis=2)
    
    return slab

def make_brucite_primitive():
    """Create Brucite primitive cell from CIF parameters."""
    # CIF Parameters
    a = 3.142
    c = 4.766
    
    # Hexagonal cell (gamma=120)
    # a1 along x
    # a2 at 120 deg
    cell = np.array([
        [a, 0.0, 0.0],
        [-0.5 * a, np.sqrt(3)/2 * a, 0.0],
        [0.0, 0.0, c]
    ])
    
    # Fractional coordinates from CIF
    # Shifted by +0.5 in Z to center the layer
    # Original: Mg(0), O(0.22, -0.22), H(0.43, -0.43)
    # Shifted: Mg(0.5), O(0.72, 0.28), H(0.93, 0.07)
    
    frac_positions = np.array([
        [0.0, 0.0, 0.5],              # Mg
        [1/3, 2/3, 0.2216 + 0.5],     # O
        [2/3, 1/3, -0.2216 + 0.5],    # O
        [1/3, 2/3, 0.4303 + 0.5],     # H
        [2/3, 1/3, -0.4303 + 0.5],    # H
    ])
    
    symbols = ['Mg', 'O', 'O', 'H', 'H']
    
    # Convert to cartesian
    positions = np.dot(frac_positions, cell)
    
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

def make_brucite(size):
    """Create Brucite slab with width ~size."""
    base = make_brucite_primitive()
    
    # Make orthogonal supercell for nicer box (rectangular)
    # Hexagonal to Orthorhombic transformation:
    # 2x1x1 supercell in hex basis is roughly rectangular?
    # Orthorhombic cell for hex:
    # a_ortho = a
    # b_ortho = sqrt(3) * a
    # 1x2x1 hex cells?
    # ASE's build.bulk can do orthorhombic=True but we are building manually.
    
    # Let's just repeat the hexagonal cell. Water packing handles non-ortho fine.
    # a = 3.14. n*3.14 ~ size
    n = max(3, int(round(size / 3.142)))
    
    # Repeat nxnx1 (single layer slab)
    # Or nxnx2 if we want thicker slab.
    slab = base.repeat((n, n, 1)) # Single layer is ~4.7A thick (including gap space)
    
    # Add vacuum
    cell = slab.get_cell()
    cell[2, 2] += max(10.0, size * 0.5)
    slab.set_cell(cell)
    slab.center(axis=2)
    
    verify_brucite_termination(slab)
    return slab

def run_benchmark():
    densities = [0.8, 0.9, 1.0, 1.1]
    sizes = [10, 20, 30]
    systems = ['PureWater', 'Periclase', 'Brucite']
    
    results_file = "benchmark_results/results.csv"
    
    print(f"Starting Benchmark...")
    print(f"Densities: {densities}")
    print(f"Sizes: {sizes}")
    print(f"Systems: {systems}")
    print("-" * 60)
    
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['System', 'Size', 'Density', 'TargetWaters', 'PlacedWaters', 'Time(s)', 'Efficiency', 'WatersPerSec'])
        
        for system_name in systems:
            for size in sizes:
                # Create system
                if system_name == 'PureWater':
                    atoms = make_pure_water(size)
                elif system_name == 'Periclase':
                    atoms = make_periclase(size)
                elif system_name == 'Brucite':
                    atoms = make_brucite(size)
                
                print(f"\nProcessing {system_name} (Size ~{size}Å)...")
                
                for density in densities:
                    print(f"  Running Density={density} g/cm³...", end="", flush=True)
                    
                    # Setup packer
                    # Use slightly relaxed constraints for higher densities/small boxes
                    packer = WaterPacker(
                        water_density=density,
                        pairwise_distances={('O', 'Mg'): 2.0, ('O', 'O'): 2.4},
                        seed=42
                    )
                    
                    try:
                        # Estimate target
                        vol = estimate_available_volume_mc(
                            atoms.get_positions(), 
                            atoms.get_chemical_symbols(), 
                            atoms.get_cell(), 
                            packer.distance_manager
                        )
                        n_target = int(vol * density * WATER_DENSITY_FACTOR)
                        n_target = max(1, n_target)
                        
                        start_time = time.time()
                        
                        # Use relaxation
                        result = pack_with_relaxation(
                            packer, 
                            atoms, 
                            n_waters=n_target,
                            max_cycles=30 if size < 15 else 20
                        )
                        
                        elapsed = time.time() - start_time
                        
                        # Calculate results
                        if system_name == 'PureWater':
                            n_placed = len(result) // 3
                        else:
                            n_placed = (len(result) - len(atoms)) // 3
                            
                        eff = (n_placed / n_target) * 100
                        rate = n_placed / elapsed
                        
                        print(f" Done. {n_placed}/{n_target} ({eff:.1f}%) in {elapsed:.2f}s")
                        
                        writer.writerow([system_name, size, density, n_target, n_placed, f"{elapsed:.4f}", f"{eff:.2f}", f"{rate:.2f}"])
                        
                        # Save structure
                        fname = f"benchmark_results/{system_name}_{size}_{density}.xyz"
                        write(fname, result, format='extxyz')
                        
                    except Exception as e:
                        print(f" Failed! {e}")
                        writer.writerow([system_name, size, density, 0, 0, 0, 0, 0])

if __name__ == "__main__":
    run_benchmark()
