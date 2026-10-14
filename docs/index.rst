pyLBA
=====

.. image:: ../logo/pylba.svg
   :width: 300
   :height: 300
   :align: center
   :alt: Two linear ballistic accumulators with animated drift samples

Simulate choices and response times with Linear Ballistic Accumulator models,
fit model parameters with PyMC, and generate predictions from fitted models.

pyLBA supports shared and accumulator-specific parameters, reproducible
simulation, and custom Bayesian priors. Its likelihood and simulator use
normal drift rates independently truncated above zero.

Start with :doc:`quickstart` for simulation, fitting, and prediction.
The :doc:`guide` explains the accumulator race and parameter definitions;
:doc:`examples` covers two-choice and multi-choice data.

.. toctree::
   :maxdepth: 2
   :titlesonly:

   installation
   quickstart
   guide
   fitting
   examples
   api
