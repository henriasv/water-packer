# Water Packer

Pack water molecules into atomic systems for molecular simulations. High-performance, O(1) collision detection using spatial hashing.

## Installation

```bash
pip install water-packer
```

For hyobj support:
```bash
pip install "water-packer[hyobj]"
```

## Quick Start

```python
from water_packer import WaterPacker
from molecular_builder import create_bulk_crystal

# Create a MgO slab
mgo = create_bulk_crystal("periclase", [15.0, 15.0, 15.0])

# Create packer and pack water
packer = WaterPacker(
    water_density=1.0,  # target density in g/cm³
    pairwise_distances={('O', 'Mg'): 2.0},
    seed=42,
)

result = packer.pack(mgo)
print(f"Added {(len(result) - len(mgo))//3} water molecules")
```

![Solvated Periclase](https://raw.githubusercontent.com/henriasv/water-packer/main/docs/_static/periclase_solvated.png)

## Features

- **Multiple input formats**: ASE Atoms, hyobj PeriodicSystem, or raw arrays
- **Species-specific distances**: Different minimum distances per atom pair
- **Automatic density calculation**: Computes number of waters from target density
- **Volume-aware packing**: Accounts for substrate exclusion volume
- **Physics Safeguards**: Warns user if requested density exceeds physical packing limits
- **Hybrid Relaxation**: Supports optional MC/minimization steps for fitting water into tight surface features

## API

### WaterPacker

```python
WaterPacker(
    min_distance=2.0,        # Default minimum distance (Å)
    pairwise_distances=None, # Species-specific: {('O', 'Mg'): 2.0}
    water_density=1.0,       # Target density (g/cm³)
    seed=None,               # Random seed
)
```

### pack_water (convenience function)

```python
from water_packer import pack_water

result = pack_water(system, n_waters=10, water_density=1.0)
```
