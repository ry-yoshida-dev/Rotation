from typing import cast

import numpy as np

from .types import FloatArray


def to_readonly_array(value: FloatArray) -> FloatArray:
    """Return a copy of ``value`` whose ``writeable`` flag is disabled."""
    readonly_value: FloatArray = np.array(value, copy=True)
    readonly_value.setflags(write=False)
    return readonly_value


def determinant(matrix: FloatArray) -> float:
    """Return the determinant of a single square matrix."""
    # NumPy types the result as Any unless the array shape is known statically.
    return cast(float, np.linalg.det(matrix))
