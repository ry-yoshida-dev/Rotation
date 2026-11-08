from typing import cast

import numpy as np
import numpy.typing as npt

from .types import FloatArray


def to_readonly_array(value: npt.ArrayLike) -> FloatArray:
    """Return a float64 copy of ``value`` whose ``writeable`` flag is disabled."""
    readonly_value: FloatArray = np.array(value, dtype=np.float64, copy=True)
    readonly_value.setflags(write=False)
    return readonly_value


def vector_norms(vectors: FloatArray, *, keepdims: bool = False) -> FloatArray:
    """Return the Euclidean norm of each vector along the last axis of ``vectors``."""
    # NumPy releases before 2.5 type the result of norm with an axis as Any.
    return cast(FloatArray, np.linalg.norm(vectors, axis=-1, keepdims=keepdims))
