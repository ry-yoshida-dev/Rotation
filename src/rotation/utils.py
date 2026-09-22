import numpy as np

from .types import FloatArray


def to_readonly_array(value: FloatArray) -> FloatArray:
    """
    Return a read-only copy of ``value``.

    Parameters
    ----------
    value: FloatArray
        The array to copy.

    Returns
    -------
    FloatArray:
        A copy of ``value`` whose ``writeable`` flag is disabled.
    """
    readonly_value: FloatArray = np.array(value, copy=True)
    readonly_value.setflags(write=False)
    return readonly_value
