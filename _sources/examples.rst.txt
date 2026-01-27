Examples
========

Example 1: Solvating a Crystal
-------------------------------

Pack water around a periclase (MgO) crystal:

.. code-block:: python

   from water_packer import WaterPacker
   from ase.build import bulk

   # Create MgO crystal
   mgo = bulk('MgO', 'rocksalt', a=4.21).repeat((3, 3, 3))

   # Pack water with Mg-O specific distance
   packer = WaterPacker(
       pairwise_distances={
           ('O', 'Mg'): 2.0,  # Water O to Mg distance
           ('O', 'O'): 2.8,   # Between MgO O and water O
       },
       water_density=1.0,
       seed=42
   )

   solvated = packer.pack(mgo, n_waters=200)

.. image:: _static/periclase_solvated.png
   :alt: Solvated Periclase Crystal
   :align: center
   :width: 600

Example 2: Reproducible Packing
--------------------------------

Generate identical structures using seeds:

.. code-block:: python

   # Run 1
   packer1 = WaterPacker(seed=123)
   result1 = packer1.pack(box, n_waters=50)

   # Run 2 - identical to run 1
   packer2 = WaterPacker(seed=123)
   result2 = packer2.pack(box, n_waters=50)

   import numpy as np
   assert np.allclose(
       result1.get_positions(),
       result2.get_positions()
   )

.. image:: _static/waterbox_solvated.png
   :alt: Reproducible Solvation Box
   :align: center
   :width: 600

Example 3: Auto-calculate Water Count
--------------------------------------

Let the packer determine how many waters fit:

.. code-block:: python

   packer = WaterPacker(water_density=1.0)
   
   # n_waters=None → use density to calculate
   result = packer.pack(substrate)
   
   print(f"Auto-packed {(len(result) - len(substrate))//3} waters")

.. image:: _static/waterbox_solvated.png
   :alt: Auto-packed Water Box
   :align: center
   :width: 600

Example 4: Complex Surfaces (Brucite)
--------------------------------------

Packing water around a Brucite (Mg(OH)2) slab with H-termination:

.. code-block:: python

   from water_packer import pack_with_relaxation
   # ... setup brucite slab ...
   
   # Use relaxation to fit waters into tight surface grooves
   result = pack_with_relaxation(packer, brucite, n_waters=32)

.. image:: _static/brucite_solvated.png
   :alt: Solvated Brucite Slab
   :align: center
   :width: 600
