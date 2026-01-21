"""
Validate water packing quality using radial distribution functions (RDF).

Tests:
1. Generate pure water boxes (orthorhombic and triclinic)
2. Calculate O-O RDF
3. Compare to expected water structure (first peak ~2.8 Å)
4. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from water_packer import WaterPacker


def calculate_rdf(positions, cell, r_max=10.0, n_bins=200):
    """
    Calculate radial distribution function with periodic boundaries.
    
    Parameters
    ----------
    positions : np.ndarray
        (N, 3) atomic positions
    cell : np.ndarray
        (3, 3) cell matrix
    r_max : float
        Maximum distance for RDF
    n_bins : int
        Number of histogram bins
    
    Returns
    -------
    r : np.ndarray
        Distance values
    g_r : np.ndarray
        RDF values
    """
    N = len(positions)
    volume = np.abs(np.linalg.det(cell))
    density = N / volume
    
    # Bin setup
    dr = r_max / n_bins
    bins = np.linspace(0, r_max, n_bins + 1)
    r = bins[:-1] + dr / 2
    
    # Calculate all pairwise distances with minimum image convention
    hist = np.zeros(n_bins)
    inv_cell = np.linalg.inv(cell)
    
    for i in range(N):
        for j in range(i + 1, N):
            # Minimum image distance
            diff = positions[i] - positions[j]
            frac_diff = np.dot(diff, inv_cell)
            frac_diff -= np.round(frac_diff)
            cart_diff = np.dot(frac_diff, cell)
            dist = np.linalg.norm(cart_diff)
            
            if dist < r_max:
                bin_idx = int(dist / dr)
                if bin_idx < n_bins:
                    hist[bin_idx] += 2  # Count both i-j and j-i
    
    # Normalize by ideal gas
    # g(r) = (count in shell) / (expected count in shell for ideal gas)
    shell_volumes = 4 * np.pi * r**2 * dr
    expected_counts = density * shell_volumes * N
    
    g_r = hist / expected_counts
    
    return r, g_r


def test_orthorhombic_box():
    """Test with standard orthorhombic box."""
    print("\n" + "="*60)
    print("Test 1: Orthorhombic Water Box (15x15x15 Å)")
    print("="*60)
    
    box = Atoms(cell=[15, 15, 15], pbc=True)
    packer = WaterPacker(water_density=1.0, seed=42)
    
    result = packer.pack(box)
    n_waters = len(result) // 3
    
    print(f"Placed {n_waters} water molecules ({len(result)} atoms)")
    print(f"Density: {n_waters / 15**3 * 18.015:.3f} g/cm³")
    
    # Calculate O-O RDF
    symbols = result.get_chemical_symbols()
    positions = result.get_positions()
    
    # Extract oxygen positions
    o_positions = positions[[i for i, s in enumerate(symbols) if s == 'O']]
    
    r, g_r = calculate_rdf(o_positions, result.get_cell(), r_max=8.0, n_bins=200)
    
    # Find first peak
    peak_idx = np.argmax(g_r[:100])  # Look in first half
    peak_r = r[peak_idx]
    peak_height = g_r[peak_idx]
    
    print(f"\nO-O RDF Analysis:")
    print(f"  First peak position: {peak_r:.2f} Å (expected ~2.8 Å for liquid water)")
    print(f"  First peak height: {peak_height:.2f}")
    
    # Check if reasonable
    if 2.5 < peak_r < 3.2:
        print(f"  ✓ First peak position is reasonable for water")
    else:
        print(f"  ⚠ First peak position unusual (check packing)")
    
    return r, g_r, "Orthorhombic 15Å"


def test_triclinic_box():
    """Test with triclinic box."""
    print("\n" + "="*60)
    print("Test 2: Triclinic Water Box")
    print("="*60)
    
    # Create triclinic cell
    cell = np.array([
        [15.0, 0.0, 0.0],
        [3.0, 15.0, 0.0],
        [2.0, 2.0, 15.0],
    ])
    
    volume = np.abs(np.linalg.det(cell))
    print(f"Cell volume: {volume:.1f} Å³")
    print(f"Cell matrix:")
    print(f"  a = [{cell[0,0]:.1f}, {cell[0,1]:.1f}, {cell[0,2]:.1f}]")
    print(f"  b = [{cell[1,0]:.1f}, {cell[1,1]:.1f}, {cell[1,2]:.1f}]")
    print(f"  c = [{cell[2,0]:.1f}, {cell[2,1]:.1f}, {cell[2,2]:.1f}]")
    
    box = Atoms(cell=cell, pbc=True)
    packer = WaterPacker(water_density=1.0, seed=42)
    
    result = packer.pack(box)
    n_waters = len(result) // 3
    
    print(f"\nPlaced {n_waters} water molecules ({len(result)} atoms)")
    print(f"Density: {n_waters * 18.015 / volume:.3f} g/cm³")
    
    # Calculate O-O RDF
    symbols = result.get_chemical_symbols()
    positions = result.get_positions()
    
    o_positions = positions[[i for i, s in enumerate(symbols) if s == 'O']]
    
    r, g_r = calculate_rdf(o_positions, result.get_cell(), r_max=8.0, n_bins=200)
    
    # Find first peak
    peak_idx = np.argmax(g_r[:100])
    peak_r = r[peak_idx]
    peak_height = g_r[peak_idx]
    
    print(f"\nO-O RDF Analysis:")
    print(f"  First peak position: {peak_r:.2f} Å")
    print(f"  First peak height: {peak_height:.2f}")
    
    if 2.5 < peak_r < 3.2:
        print(f"  ✓ RDF looks reasonable for triclinic cell")
    else:
        print(f"  ⚠ RDF unusual - may indicate issues with PBC handling")
    
    return r, g_r, "Triclinic"


def test_large_box():
    """Test with larger box for better statistics."""
    print("\n" + "="*60)
    print("Test 3: Large Box (20x20x20 Å) for Better Statistics")
    print("="*60)
    
    box = Atoms(cell=[20, 20, 20], pbc=True)
    packer = WaterPacker(water_density=1.0, seed=42)
    
    result = packer.pack(box)
    n_waters = len(result) // 3
    
    print(f"Placed {n_waters} water molecules")
    
    # Calculate RDF
    symbols = result.get_chemical_symbols()
    positions = result.get_positions()
    o_positions = positions[[i for i, s in enumerate(symbols) if s == 'O']]
    
    r, g_r = calculate_rdf(o_positions, result.get_cell(), r_max=10.0, n_bins=250)
    
    peak_idx = np.argmax(g_r[:125])
    peak_r = r[peak_idx]
    
    print(f"\nO-O RDF First peak: {peak_r:.2f} Å")
    
    return r, g_r, "Large 20Å"


def plot_rdfs(rdf_data):
    """Plot all RDFs together."""
    print("\n" + "="*60)
    print("Creating RDF comparison plot...")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for r, g_r, label in rdf_data:
        ax.plot(r, g_r, label=label, linewidth=2)
    
    # Mark expected first peak for water
    ax.axvline(2.8, color='gray', linestyle='--', alpha=0.5, label='Expected peak ~2.8Å')
    
    ax.set_xlabel('r (Å)', fontsize=12)
    ax.set_ylabel('g(r)', fontsize=12)
    ax.set_title('O-O Radial Distribution Function for Packed Water', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig('/Users/henriasv/repos/antigravity_repos/water_packer/rdf_validation.png', dpi=150)
    print("✓ Saved plot to rdf_validation.png")
    
    return fig


if __name__ == "__main__":
    print("\nWater Packing Quality Validation")
    print("="*60)
    print("Testing RDF and triclinic boxes...")
    
    rdf_data = []
    
    # Test 1: Orthorhombic
    r1, g1, l1 = test_orthorhombic_box()
    rdf_data.append((r1, g1, l1))
    
    # Test 2: Triclinic
    r2, g2, l2 = test_triclinic_box()
    rdf_data.append((r2, g2, l2))
    
    # Test 3: Large box
    r3, g3, l3 = test_large_box()
    rdf_data.append((r3, g3, l3))
    
    # Plot comparison
    plot_rdfs(rdf_data)
    
    print("\n" + "="*60)
    print("✓ Validation complete!")
    print("="*60)
