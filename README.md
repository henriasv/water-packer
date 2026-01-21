# Water Packer

Pack water molecules into atomic systems for molecular simulations.

## Installation

```bash
pip install -e .
```

For hyobj support:
```bash
pip install -e ".[hyobj]"
```

## Quick Start

```python
from water_packer import WaterPacker
from ase.io import read

# Load a structure
mgo = read('periclase.cif')

# Create packer with custom distance constraints
packer = WaterPacker(
    min_distance=2.0,  # Default fallback distance
    pairwise_distances={('O', 'Mg'): 2.5},  # Species-specific
    water_density=1.0,  # g/cm³
    seed=42,  # For reproducibility
)

# Pack water
result = packer.pack(mgo)
print(f"Added {len(result) - len(mgo)} atoms")
```

## Features

- **Multiple input formats**: ASE Atoms, hyobj PeriodicSystem, or raw arrays
- **Species-specific distances**: Different minimum distances per atom pair
- **Automatic density calculation**: Computes number of waters from target density
- **Volume-aware packing**: Accounts for substrate exclusion volume

## API

### WaterPacker

```python
WaterPacker(
    min_distance=2.0,        # Default minimum distance (Å)
    pairwise_distances=None, # Species-specific: {('O', 'Mg'): 2.5}
    water_density=1.0,       # Target density (g/cm³)
    seed=None,               # Random seed
)
```

### pack_water (convenience function)

```python
from water_packer import pack_water

result = pack_water(system, n_waters=10, water_density=1.0)
```
