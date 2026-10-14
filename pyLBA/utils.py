"""Utilities for parameter broadcasting and response data."""

import numpy as np
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable
from .core.parameters import Parameter


def infer_n_acc(parameters, response=None, n_acc=None):
    """Infer race size from parameter vectors and observed choice labels."""
    lengths = []
    for value in parameters.to_dict().values():
        if isinstance(value, TensorVariable):
            length = value.type.shape[0] if value.ndim == 1 else None
        else:
            array = np.asarray(value)
            length = array.size if array.ndim == 1 else None
        if length is not None:
            lengths.append(length)
    observed = (
        int(np.max(response)) + 1 if response is not None and len(response) else 1
    )
    if n_acc is None:
        n_acc = max(lengths + [observed])
    if not isinstance(n_acc, (int, np.integer)) or n_acc < 1:
        raise ValueError("n_acc must be a positive integer")
    if observed > n_acc or any(length != n_acc for length in lengths):
        raise ValueError("Parameter vectors and response labels must match n_acc")
    return int(n_acc)


def broadcast_parameters(n, *args):
    """Broadcast numeric or symbolic scalars to vectors of length ``n``."""
    result = []
    for value in args:
        value = value.value if isinstance(value, Parameter) else value
        symbolic = isinstance(value, TensorVariable)
        value = value if symbolic else np.asarray(value, dtype=float)
        if value.ndim == 0:
            value = pt.repeat(value, n) if symbolic else np.repeat(value, n)
        elif value.ndim != 1:
            raise ValueError("Parameters must be scalar or one-dimensional")
        if symbolic:
            value = pt.specify_shape(value, (n,))
        elif len(value) != n:
            raise ValueError(
                f"Expected parameter vector of length {n}, got {len(value)}"
            )
        result.append(value)
    return tuple(result)


def validate_data(data):
    """Require finite positive times and non-negative integer response labels."""
    if not all(column in data.columns for column in ("rt", "response")):
        raise ValueError("Data must contain columns: ['rt', 'response']")
    if data.empty:
        raise ValueError("Data must contain at least one trial")
    rt, response = data["rt"].to_numpy(), data["response"].to_numpy()
    if (
        not np.issubdtype(rt.dtype, np.number)
        or not np.isfinite(rt).all()
        or (rt <= 0).any()
    ):
        raise ValueError("All reaction times must be positive and finite")
    if not np.issubdtype(response.dtype, np.number) or not np.isfinite(response).all():
        raise ValueError("All responses must be finite non-negative integers")
    if (response < 0).any() or (response != np.floor(response)).any():
        raise ValueError("All responses must be non-negative integers")


def check_convergence(trace, r_hat_threshold=1.01):
    """Return whether all reported R-hat values are finite and below a limit."""
    import arviz as az

    r_hat = az.summary(trace)["r_hat"]
    return bool(np.isfinite(r_hat).all() and (r_hat < r_hat_threshold).all())
