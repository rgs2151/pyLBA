"""Bayesian fitting for accumulator models."""

from abc import ABC, abstractmethod
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from .utils import validate_data


class BaseFitter(ABC):
    """Base interface for model fitters."""

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def fit(self, data, **kwargs):
        """Fit a model to response-time and choice data."""


class MCMCFitter(BaseFitter):
    """MCMC fitting using PyMC."""

    def fit(
        self,
        data,
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        progressbar=True,
        priors=None,
        n_acc=None,
        **kwargs,
    ):
        """Sample the posterior and return ArviZ InferenceData.

        ``priors`` maps parameter names to fixed numeric values or callables
        ``(name, n_acc)`` creating PyMC variables inside the fitting model.
        ``n_acc`` defaults to the largest response label plus one. Set it
        explicitly when higher-numbered choices are absent from the data.
        Remaining keyword arguments are forwarded to ``pymc.sample``.
        """
        validate_data(data)
        rt = data["rt"].to_numpy(dtype=float)
        response = data["response"].to_numpy(dtype="int64")
        observed = int(response.max()) + 1
        if n_acc is None:
            n_acc = observed
        if not isinstance(n_acc, (int, np.integer)) or n_acc < observed:
            raise ValueError("n_acc must include every observed response label")
        if kwargs.get("return_inferencedata") is False:
            raise ValueError("Fitting requires return_inferencedata=True")
        with pm.Model():
            values = self.model.get_default_priors(
                n_acc, priors=priors, rt_min=float(rt.min())
            )
            parameters = self.model.get_parameter_class()(**values)
            p = self.model.pdf(rt, response, parameters, n_acc=n_acc)
            pm.Potential("likelihood", pt.sum(pt.log(p)))
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                progressbar=progressbar,
                **kwargs,
            )
        trace.attrs["n_acc"] = int(n_acc)
        return trace


class EMFitter(BaseFitter):
    """Expectation-Maximization interface."""

    def fit(self, data, **kwargs):
        """Raise NotImplementedError; EM fitting is not implemented."""
        raise NotImplementedError("EM fitting not yet implemented")
