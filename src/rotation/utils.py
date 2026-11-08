import numpy as np
import numpy.typing as npt

from .types import FloatArray


def to_readonly_array(value: npt.ArrayLike) -> FloatArray:
    """Return a float64 copy of ``value`` whose ``writeable`` flag is disabled."""
    readonly_value: FloatArray = np.array(value, dtype=np.float64, copy=True)
    readonly_value.setflags(write=False)
    return readonly_value
