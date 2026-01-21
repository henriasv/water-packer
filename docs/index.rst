water-packer
=============

Pure-Python water molecule packing for molecular simulations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api
   examples

Features
--------

* **Pure Python**: No external binaries required
* **Multiple input formats**: ASE Atoms, hyobj PeriodicSystem, or raw arrays  
* **Species-specific distances**: Fine-grained control over atom pair distances
* **Automatic density calculation**: Computes water count from target density
* **Reproducible**: Seeded random generation for exact reproducibility
* **Fast**: Spatial hash grid for O(1) collision detection

Installation
------------

.. code-block:: bash

   pip install water-packer

Quick Start
-----------

.. code-block:: python

   from water_packer import WaterPacker
   from ase.io import read

   # Load substrate
   substrate = read('structure.cif')

   # Pack water
   packer = WaterPacker(
       pairwise_distances={('O', 'Mg'): 2.5},
       water_density=1.0,
       seed=42
   )
   result = packer.pack(substrate, n_waters=100)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
