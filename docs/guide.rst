Model guide
===========

The accumulator race
--------------------

Each accumulator begins at a point drawn uniformly between zero and ``A``.
Its drift is drawn from a normal distribution with mean ``v`` and standard
deviation ``s``, independently truncated above zero. Evidence increases at
that constant drift until it reaches threshold ``b``.

The finish time is ``tau + (b - start) / drift``. The first accumulator to
finish determines both the response and its response time.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 15 55 30

   * - Parameter
     - Meaning
     - Domain
   * - ``A``
     - Upper bound of the uniform starting point
     - Positive
   * - ``b``
     - Response threshold
     - Greater than ``A``
   * - ``v``
     - Mean of the underlying, untruncated normal drift
     - Finite
   * - ``s``
     - Standard deviation of the underlying normal drift
     - Positive
   * - ``tau``
     - Non-decision time
     - Non-negative

Times and non-decision times must use the same units. ``v`` is the mean of
the underlying normal distribution, not the mean after truncation.

Working with parameters
-----------------------

Parameter fields accept scalars or one-dimensional vectors. Scalars broadcast
across the race; vectors must match its accumulator count.

.. code-block:: python

   from pyLBA import LBAParameters

   parameters = LBAParameters(A=1, b=[2, 3], v=2, s=1, tau=0.1)
   print(parameters.validate())
   values = parameters.to_dict()
   restored = LBAParameters.from_dict(values)
   thresholds = parameters.b.value

``validate`` checks numeric parameter values and shapes. Simulation rejects
invalid parameters. During fitting, parameter objects can also carry symbolic
PyTensor variables.

Data format
-----------

Input data is a nonempty pandas DataFrame with finite positive ``rt`` values
and non-negative integer ``response`` labels. Choices are numbered from zero.

.. code-block:: python

   import pandas as pd

   data = pd.DataFrame({
       "rt": [0.5, 0.7, 0.4, 0.9],
       "response": [0, 1, 0, 1],
   })

Fitting infers the number of accumulators from the largest response label plus
one. Pass ``n_acc`` when the highest-numbered choices have no observations.

Likelihood
----------

``LBAModel.pdf`` returns a PyTensor expression for the joint density of response
time and choice. Each first-passage density and CDF is normalized by the
probability of a positive drift. The winning density is multiplied by the
survival probabilities of the other accumulators. This matches simulation's
independent positive truncation.

.. code-block:: python

   from pyLBA import LBAModel

   model = LBAModel()
   density = model.pdf(
       data.rt.to_numpy(), data.response.to_numpy(), parameters, n_acc=2
   ).eval()

The result is a density, not a log likelihood. It is zero outside the model's
parameter or decision-time support.

Reference
---------

Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice
response time: Linear ballistic accumulation. *Cognitive Psychology*,
57(3), 153–178.
