"""Linear Ballistic Accumulator with independently positive-truncated drifts."""

from dataclasses import dataclass
from typing import Union
import numpy as np
import pandas as pd
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable
from ..core.accumulator import AccumulatorModel
from ..core.parameters import ModelParameters
from ..utils import broadcast_parameters, infer_n_acc, validate_data


@dataclass(frozen=True)
class LBAParameters(ModelParameters):
    """Starting-point bound, threshold, drift mean/SD, and non-decision time."""

    A: Union[float, np.ndarray]
    b: Union[float, np.ndarray]
    v: Union[float, np.ndarray]
    s: Union[float, np.ndarray]
    tau: Union[float, np.ndarray]

    def validate(self):
        """Check numeric values have compatible shapes and valid LBA support."""
        if any(isinstance(value, TensorVariable) for value in self.to_dict().values()):
            raise TypeError("validate() requires numeric parameters")
        try:
            n = infer_n_acc(self)
            A, b, v, s, tau = broadcast_parameters(
                n, self.A, self.b, self.v, self.s, self.tau
            )
            return bool(
                all(np.isfinite(x).all() for x in (A, b, v, s, tau))
                and (A > 0).all()
                and (b > A).all()
                and (s > 0).all()
                and (tau >= 0).all()
            )
        except ValueError:
            return False


class LBAModel(AccumulatorModel):
    """An LBA race with normal drifts independently truncated above zero.

    Each accumulator has a uniform starting point and constant drift per trial.
    The first accumulator to reach its threshold determines the response.
    """

    def __init__(self):
        super().__init__("Linear Ballistic Accumulator")

    def generate_data(self, n_trials, parameters, n_acc=None, seed=None):
        """Simulate response times and choices with an optional random seed."""
        from ..synthetic import generate_LBA

        return generate_LBA(n_trials, parameters, n_acc=n_acc, seed=seed)

    def pdf(self, rt, response, parameters, n_acc=None):
        """Return a symbolic joint density of response time and choice.

        Scalars broadcast across accumulators. Vector parameters must match
        ``n_acc``. Density is zero outside parameter and response-time support.
        """
        rt = np.asarray(rt, dtype=float)
        response = np.asarray(response)
        if rt.ndim != 1 or response.ndim != 1 or rt.shape != response.shape:
            raise ValueError(
                "rt and response must be one-dimensional arrays of equal length"
            )
        validate_data(pd.DataFrame({"rt": rt, "response": response}))
        response = response.astype("int64")
        n_acc = infer_n_acc(parameters, response, n_acc)
        A, b, v, s, tau = [
            pt.as_tensor_variable(x)
            for x in broadcast_parameters(
                n_acc,
                parameters.A,
                parameters.b,
                parameters.v,
                parameters.s,
                parameters.tau,
            )
        ]
        valid = pt.all((A > 0) & (b > A) & (s > 0) & (tau >= 0))
        # Safe intermediates keep derivatives finite outside support.
        safe_A, safe_s = pt.maximum(A, 1e-12), pt.maximum(s, 1e-12)
        t = rt[None, :] - tau[:, None]
        safe_t = pt.maximum(t, 1e-8)
        g = (b[:, None] - safe_A[:, None] - safe_t * v[:, None]) / (
            safe_t * safe_s[:, None]
        )
        h = (b[:, None] - safe_t * v[:, None]) / (safe_t * safe_s[:, None])
        cdf = lambda x: 0.5 * pt.erfc(-x / np.sqrt(2.0))
        density = lambda x: pt.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
        positive_mass = pt.maximum(cdf(v / safe_s), 1e-300)[:, None]
        f = (
            v[:, None] * (cdf(h) - cdf(g)) + safe_s[:, None] * (density(g) - density(h))
        ) / safe_A[:, None]
        F = (
            1
            + ((b - safe_A)[:, None] - safe_t * v[:, None]) / safe_A[:, None] * cdf(g)
            - (b[:, None] - safe_t * v[:, None]) / safe_A[:, None] * cdf(h)
            + safe_t * safe_s[:, None] / safe_A[:, None] * (density(g) - density(h))
        )
        f = pt.switch(t > 0, pt.maximum(f / positive_mass, 0), 0)
        F = pt.switch(t > 0, pt.clip(F / positive_mass, 0, 1), 0)
        survival = pt.where(np.arange(n_acc)[:, None] == response[None, :], 1, 1 - F)
        joint = f[response, np.arange(len(rt))] * pt.prod(survival, axis=0)
        return pt.switch(valid, joint, 0)

    def get_parameter_class(self):
        """Return :class:`LBAParameters`."""
        return LBAParameters

    def get_default_priors(self, n_acc, priors=None, rt_min=1.0):
        """Create priors in the active PyMC model, with optional overrides.

        Overrides are fixed numeric values or callables ``(name, n_acc)``
        returning a PyMC variable. Defaults use positive A, b=A+gap, unit
        drift SD, normal drift means, and tau below the shortest observed RT.
        """
        import pymc as pm

        priors = dict(priors or {})
        unknown = set(priors) - {"A", "b", "v", "s", "tau"}
        if unknown:
            raise ValueError(f"Unknown prior parameters: {sorted(unknown)}")
        values = {}
        for name in ("A", "b", "v", "s", "tau"):
            if name in priors:
                value = priors[name]
                value = value(name, n_acc) if callable(value) else value
                if isinstance(value, TensorVariable):
                    if value.name != name or not any(
                        value is variable
                        for variable in pm.modelcontext(None).named_vars.values()
                    ):
                        raise ValueError(
                            "Use a prior callable to create a named variable in the fitting model"
                        )
                    values[name] = value
                else:
                    values[name] = pm.Deterministic(
                        name,
                        pt.as_tensor_variable(broadcast_parameters(n_acc, value)[0]),
                    )
            elif name == "A":
                values[name] = pm.HalfNormal(name, sigma=2, shape=n_acc)
            elif name == "b":
                values[name] = pm.Deterministic(
                    name, values["A"] + pm.HalfNormal("b_gap", sigma=2, shape=n_acc)
                )
            elif name == "v":
                values[name] = pm.Normal(name, mu=2, sigma=1, shape=n_acc)
            elif name == "s":
                values[name] = pm.Deterministic(name, pt.ones(n_acc))
            else:
                values[name] = pm.Uniform(name, lower=0, upper=rt_min, shape=n_acc)
        return values
