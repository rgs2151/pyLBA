"""Numeric and symbolic model parameters."""

from dataclasses import dataclass, fields
import numpy as np
from pytensor.tensor.variable import TensorVariable


class Parameter:
    """A scalar or vector parameter, including symbolic PyTensor values."""

    def __init__(self, name, value):
        self.name = name
        if isinstance(value, TensorVariable):
            self.value = value
        else:
            try:
                array = np.asarray(value, dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Parameter '{name}' must be numeric or symbolic"
                ) from exc
            self.value = float(array) if array.ndim == 0 else array
        if getattr(self.value, "ndim", 0) > 1:
            raise ValueError(f"Parameter '{name}' must be scalar or one-dimensional")

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self.value, dtype=dtype)
        return array.copy() if copy else array

    def __float__(self):
        return float(self.value)

    def __eq__(self, other):
        return self.value == (other.value if isinstance(other, Parameter) else other)

    def __repr__(self):
        return repr(self.value)


@dataclass(frozen=True)
class ModelParameters:
    """Base dataclass for named scalar or vector parameters."""

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, Parameter):
                object.__setattr__(self, field.name, Parameter(field.name, value))

    def to_dict(self):
        """Return unwrapped parameter values, keyed by field name."""
        return {field.name: getattr(self, field.name).value for field in fields(self)}

    @classmethod
    def from_dict(cls, values):
        """Create parameters from a mapping of field names to values."""
        return cls(**values)
