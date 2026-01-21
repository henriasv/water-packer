"""
Generate solvated Periclase (MgO) and Brucite (Mg(OH)2) examples for visualization.
Exports to .extxyz format.
"""

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write
from water_packer import WaterPacker, pack_with_relaxation

def create_periclase():
    """Create a Periclase (MgO) slab."""
    print("Creating Periclase (MgO) slab...")
    # Standard rocksalt structure - use cubic cell for easy slab making
    fatoms = bulk('MgO', 'rocksalt', a=4.21, cubic=True)
    
    # Replicate to make a decent size slab (approx 20x20 Å)
    # 5x5x3 unit cells
    atoms = fatoms.repeat((5, 5, 3))
    
    # Add vacuum for water (15 Å)
    cell = atoms.get_cell()
    print(f"  Initial size: {cell[0,0]:.2f} x {cell[1,1]:.2f} x {cell[2,2]:.2f} Å")
    
    cell[2, 2] += 15.0
    atoms.set_cell(cell)
    atoms.center(axis=2)
    
    return atoms

def create_brucite():
    """Create Brucite (Mg(OH)2) slab from CIF parameters with H-termination."""
    print("Creating Brucite (Mg(OH)2) slab...")
    
    # CIF Parameters
    a = 3.142
    c = 4.766
    
    # Hexagonal cell (gamma=120)
    cell = np.array([
        [a, 0.0, 0.0],
        [-0.5 * a, np.sqrt(3)/2 * a, 0.0],
        [0.0, 0.0, c]
    ])
    
    # Fractional coordinates from CIF (shifted +0.5 Z for H-termination)
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
    
    base = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    
    # Replicate to make larger slab
    # 3x3x2
    atoms = base.repeat((3, 3, 2))
    
    # Add vacuum
    cell = atoms.get_cell()
    print(f"  Initial size: {cell[0,0]:.2f} x {cell[1,1]:.2f} x {cell[2,2]:.2f} Å(z)")
    
    cell[2, 2] += 15.0
    atoms.set_cell(cell)
    atoms.center(axis=2)
    
    return atoms

def solvate_system(name, atoms, density=1.0):
    """Pack water into the system and save."""
    print(f"\nSolvating {name}...")
    
    # Use realistic constraints
    # reduce O-Mg distance slightly to allow solvation layers
    packer = WaterPacker(
        water_density=density,
        pairwise_distances={('O', 'Mg'): 2.0},
        seed=42
    )
    
    # Use hybrid relaxation for better packing near surfaces
    # Target number of waters calculated automatically by packing function if n_waters is None
    # But pack_with_relaxation needs explicit n_waters or we let manual calculation
    
    # First, let's just use standard pack to estimate volume and count
    # But pack_with_relaxation takes packer and system, NOT n_waters=None usually?
    # Actually, let's use the packer.pack() first to get a "standard" result,
    # OR calculate n_waters from available volume.
    
    # Let's use the optimized packer first as it's fast
    # Ideally we'd use relaxation if standard packing struggles
    
    try:
        # Generate number of waters based on volume
        from water_packer.volume_estimation import estimate_available_volume_mc
        from water_packer.constants import WATER_DENSITY_FACTOR
        
        vol = estimate_available_volume_mc(
            atoms.get_positions(), 
            atoms.get_chemical_symbols(), 
            atoms.get_cell(), 
            packer.distance_manager
        )
        n_waters = int(vol * density * WATER_DENSITY_FACTOR)
        print(f"  Estimated available volume: {vol:.1f} Å³")
        print(f"  Target water molecules: {n_waters}")
        
        # Use hybrid packing
        result = pack_with_relaxation(
            packer, 
            atoms, 
            n_waters=n_waters,
            max_cycles=20  # Give it some effort
        )
        
        n_placed = (len(result) - len(atoms)) // 3
        print(f"  ✓ Placed {n_placed}/{n_waters} waters")
        
        # Save trajectory/structure
        filename = f"{name.lower()}_solvated.extxyz"
        write(filename, result, format='extxyz')
        print(f"  ✓ Saved structure to {filename}")

        # Generate PNG for documentation
        try:
            import matplotlib.pyplot as plt
            from ase.visualize.plot import plot_atoms
            
            # Create a nice plot
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Rotate for better view
            result.rotate(10, 'x')
            result.rotate(10, 'y')
            
            # Plot
            plot_atoms(result, ax, radii=0.8, rotation=('10x,10y,0z'))
            ax.set_axis_off()
            ax.set_title(f"Solvated {name}", fontsize=16)
            
            img_filename = f"docs/_static/{name.lower()}_solvated.png"
            fig.savefig(img_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Saved image to {img_filename}")
            return filename, img_filename

        except Exception as e:
            print(f"  ⚠ Could not generate image: {e}")
            return filename, None
        
    except Exception as e:
        print(f"  ❌ Error solvating {name}: {e}")
        return None, None

if __name__ == "__main__":
    generated_files = []
    
    # 1. Periclase
    periclase = create_periclase()
    f1, img1 = solvate_system("Periclase", periclase)
    if f1: generated_files.append(f1)
    if img1: generated_files.append(img1)
    
    # 2. Brucite
    brucite = create_brucite()
    f2, img2 = solvate_system("Brucite", brucite)
    if f2: generated_files.append(f2)
    if img2: generated_files.append(img2)
    
    print("\nSummary:")
    for f in generated_files:
        print(f"- {f}")
