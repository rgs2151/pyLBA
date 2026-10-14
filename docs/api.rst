API reference
=============

Reference for simulation, fitting, parameters, and response-data utilities.

LBA model
---------

.. autoclass:: pyLBA.models.linear_ballistic_accumulator.LBAParameters
   :members:

.. autoclass:: pyLBA.models.linear_ballistic_accumulator.LBAModel
   :members:

Simulation
----------

.. autofunction:: pyLBA.synthetic.generate_LBA

Core interfaces
---------------

Import core classes from ``pyLBA`` or ``pyLBA.core``.

.. autoclass:: pyLBA.core.parameters.Parameter
   :members:

.. autoclass:: pyLBA.core.parameters.ModelParameters
   :members:

.. autoclass:: pyLBA.core.accumulator.AccumulatorModel
   :members:

Fitters
-------

.. autoclass:: pyLBA.fitting.MCMCFitter
   :members:

.. autoclass:: pyLBA.fitting.EMFitter
   :members:

Utilities
---------

.. automodule:: pyLBA.utils
   :members:
