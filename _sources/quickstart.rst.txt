Quick Start
===========

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install water-packer

Or from source:

.. code-block:: bash

   git clone https://github.com/henriasv/water-packer
   cd water-packer
   pip install -e .

Basic Usage
-----------

Packing water into an empty box:

.. code-block:: python

   from water_packer import WaterPacker
   from ase import Atoms

   # Create empty box
   box = Atoms(cell=[20, 20, 20], pbc=True)

   # Pack water
   packer = WaterPacker(seed=42)
   result = packer.pack(box, n_waters=100)
   
   print(f"Packed {len(result)//3} water molecules")

Packing around a substrate:

.. code-block:: python

   from water_packer import WaterPacker
   from ase.io import read

   # Load substrate
   substrate = read('MgO.cif')

   # Pack with species-specific distances
   packer = WaterPacker(
       pairwise_distances={('O', 'Mg'): 2.5},
       water_density=1.0,
       seed=42
   )
   
   result = packer.pack(substrate, n_waters=200)

Automatic water count from density:

.. code-block:: python

   # Let packer auto-calculate water count
   result = packer.pack(substrate)  # n_waters=None → auto-calc
