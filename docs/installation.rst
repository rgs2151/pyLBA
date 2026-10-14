Installation
============

pyLBA requires Python 3.10 or newer. It uses NumPy and pandas for data,
PyMC and PyTensor for Bayesian estimation, and ArviZ for posterior summaries.

Install from GitHub
-------------------

.. code-block:: bash

   python -m pip install git+https://github.com/rgs2151/pyLBA.git

Verify the installation:

.. code-block:: python

   import pyLBA
   print(pyLBA.__version__)

Editable installation
---------------------

For local development or running the example notebook:

.. code-block:: bash

   git clone https://github.com/rgs2151/pyLBA.git
   cd pyLBA
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e '.[dev,docs]'

On Windows, activate the environment with ``.venv\Scripts\activate`` instead.

Build the documentation
-----------------------

From the repository root:

.. code-block:: bash

   python -m pip install -e '.[docs]'
   python -m sphinx -W --keep-going -b html docs docs/_build/html
   python -m http.server 8000 --directory docs/_build/html

Open ``http://localhost:8000`` to browse the generated site. Documentation builds
render the guides and API reference without running simulations or MCMC.
