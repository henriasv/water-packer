"""Check bond lengths in generated extxyz files."""
from ase.io import read
import numpy as np

def check_file(filename):
    print(f"\nChecking {filename}...")
    atoms = read(filename)
    
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Waters are appended at the end usually
    # But let's find O atoms and their nearest H neighbors
    
    o_indices = [i for i, s in enumerate(symbols) if s == 'O']
    h_indices = [i for i, s in enumerate(symbols) if s == 'H']
    
    print(f"  Total atoms: {len(atoms)}")
    print(f"  Species present: {set(symbols)}")
    print(f"  Found {len(o_indices)} Oxygens and {len(h_indices)} Hydrogens")
    
    bad_bonds = 0
    max_bond = 0.0
    
    # Assuming water molecules are contiguous or we search for them
    # Since we packed them as triplets, they should be triplets in the list order
    # usually: Substrate... then O H H, O H H...
    
    # Let's try to identify triplets
    # Iterate through atoms, looking for O followed by 2 H
    # Note: Substrate might have Os and Hs too (Brucite has Mg, O, H)
    # So we need to be careful.
    
    # Let's assume the packing adds waters at the end.
    # We can check the last N atoms where N = n_waters * 3
    
    n_waters = 0
    
    # Scan from end
    i = len(atoms) - 1
    while i >= 2:
        if symbols[i] == 'H' and symbols[i-1] == 'H' and symbols[i-2] == 'O':
            # It's a water candidate
            o_pos = positions[i-2]
            h1_pos = positions[i-1]
            h2_pos = positions[i]
            
            # Check bond lengths WITHOUT PBC (checking if contiguous in file)
            d1 = np.linalg.norm(h1_pos - o_pos)
            d2 = np.linalg.norm(h2_pos - o_pos)
            
            max_bond = max(max_bond, d1, d2)
            
            if d1 > 1.5 or d2 > 1.5:
                print(f"  ❌ Bad water at indices {i-2},{i-1},{i}")
                print(f"     O:  {o_pos}")
                print(f"     H1: {h1_pos} (dist: {d1:.2f})")
                print(f"     H2: {h2_pos} (dist: {d2:.2f})")
                bad_bonds += 1
            
            n_waters += 1
            i -= 3
        else:
            i -= 1
            
    print(f"  Checked {n_waters} distinct water triplets (O-H-H pattern).")
    print(f"  Max O-H bond length found: {max_bond:.4f} Å")
    
    if bad_bonds == 0:
        print("  ✓ All water molecules are contiguous in Cartesian space.")
    else:
        print(f"  ❌ Found {bad_bonds} broken water molecules.")

if __name__ == "__main__":
    check_file("periclase_solvated.extxyz")
    # check_file("brucite_solvated.extxyz")  # Brucite substrate has H, confuses simple check
