"""
Generic synthetic data generator for accumulator models.
"""

import numpy as np
import pandas as pd
from .utils import broadcast_parameters, infer_n_acc
from .models.linear_ballistic_accumulator import LBAParameters


def generate_LBA(
    n_trials: int, parameters: LBAParameters, n_acc: int = None, seed: int = None
) -> pd.DataFrame:
    """
    Generate synthetic response time and choice data for an accumulator model.

    Parameters
    ----------
    n_trials : int
        Number of trials to simulate.
    parameters : LBAParameters
        Model parameters with fields A, b, v, s, tau.
    n_acc : int, optional
        Number of accumulators.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'rt' and 'response'.
    """
    if not isinstance(n_trials, (int, np.integer)) or n_trials < 0:
        raise ValueError("n_trials must be a non-negative integer")
    if not parameters.validate():
        raise ValueError(
            "Require finite parameters with A > 0, b > A, s > 0 and tau >= 0"
        )
    n_acc = infer_n_acc(parameters, n_acc=n_acc)
    rng = np.random.default_rng(seed)

    # Broadcast parameters to shape [n_acc]
    A_arr, b_arr, v_arr, s_arr, tau_arr = broadcast_parameters(
        n_acc, parameters.A, parameters.b, parameters.v, parameters.s, parameters.tau
    )
    # Uniform start-points
    k = rng.random((n_acc, n_trials)) * A_arr[:, None]

    from scipy.stats import truncnorm

    d = truncnorm.rvs(
        -v_arr[:, None] / s_arr[:, None],
        np.inf,
        loc=v_arr[:, None],
        scale=s_arr[:, None],
        size=(n_acc, n_trials),
        random_state=rng,
    )

    # Finish times
    T = tau_arr[:, None] + (b_arr[:, None] - k) / d

    # Extract RT and response
    rt = T.min(axis=0)
    resp = T.argmin(axis=0)

    return pd.DataFrame({"rt": rt.astype("float64"), "response": resp.astype("int32")})
