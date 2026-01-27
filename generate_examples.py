"""
Generate solvated Periclase (MgO) and Brucite (Mg(OH)2) examples for visualization.
Exports to .extxyz format.
"""

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write
from water_packer import WaterPacker, pack_with_relaxation

from molecular_builder import create_bulk_crystal

def create_periclase():
    """Create a Periclase (MgO) slab using molecular-builder."""
    print("Creating Periclase (MgO) slab...")
    
    # Create bulk unit crystal ([1,1,1] means 1 unit cell? No, size argument)
    # create_bulk_crystal returns an ase.Atoms object
    # For cubic, size is usually array like [a, b, c] dimensions or repeat?
    # Checking molecular_builder.core:
    #   crystal = crystals[name]
    #   atoms = ase.spacegroup.crystal(..., cellpar=[a,b,c, ...])
    #   atoms = atoms.repeat(np.ceil(size / atoms.cell.lengths())) ?
    # Actually create_bulk_crystal implementation:
    #   ... creates crystal ...
    #   if size is not None: repeats
    # But size is in Angstroms? "size: size of the bulk crystal"
    # Let's provide a size that ensures we get enough repeats.
    # Periclase a=4.21. 5*4.21=21.05.
    
    atoms = create_bulk_crystal("periclase", [22.0, 22.0, 13.0])
    
    # Add vacuum for water (15 Å)
    cell = atoms.get_cell()
    print(f"  Bulky size: {cell[0,0]:.2f} x {cell[1,1]:.2f} x {cell[2,2]:.2f} Å")
    
    cell[2, 2] += 15.0
    atoms.set_cell(cell)
    atoms.center(axis=2)
    
    return atoms

def create_brucite():
    """Create Brucite (Mg(OH)2) slab using molecular-builder."""
    print("Creating Brucite (Mg(OH)2) slab...")
    
    # Brucite a=3.14, c=4.77.
    # 3x3x2 repeats -> ~9.4 x 9.4 x 9.5
    # Warning: create_bulk_crystal takes size in Angstroms?
    # Let's ask for 10x10x10 to be safe
    atoms = create_bulk_crystal("brucite", [10.0, 10.0, 10.0])
    
    # Add vacuum
    cell = atoms.get_cell()
    print(f"  Bulk size: {cell[0,0]:.2f} x {cell[1,1]:.2f} x {cell[2,2]:.2f} Å(z)")
    
    # User warning: Brucite cutting might misplace OH.
    # The crystal comes from ase.spacegroup.crystal.
    # center(axis=2) should handle the vacuum placement safely AROUND the atoms.
    
    cell[2, 2] += 15.0
    atoms.set_cell(cell)
    atoms.center(axis=2) # This is key
    
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
            
            # Rotate for better view (side view)
            # result.rotate(90, 'x') # Actually plot_atoms rotation is easier to control
            
            # Plot
            # 90x rotates so z-axis is vertical? No, 90x rotates around x. 
            # If slab is in XY plane (Z normal), we want to see it from the side (along X or Y).
            # Default view is Z coming out of screen (XY plane).
            # -90x rotates Z to Y (up). So we see X-Z plane.
            plot_atoms(result, ax, radii=0.8, rotation=('-90x,0y,0z'))
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

def create_water_box():
    """Create a simple empty box for water packing examples."""
    print("Creating empty box for water packing...")
    # 15x15x15 Å box
    a = 15.0
    cell = np.eye(3) * a
    atoms = Atoms(cell=cell, pbc=True)
    return atoms

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
    
    # 3. Generic Box (for reproducible & auto examples)
    water_box = create_water_box()
    f3, img3 = solvate_system("WaterBox", water_box, density=1.0)
    if f3: generated_files.append(f3)
    if img3: generated_files.append(img3)

    print("\nSummary:")
    for f in generated_files:
        print(f"- {f}")


