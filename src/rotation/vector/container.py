from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..types import FloatArray
from ..utils import to_readonly_array
from .mixin.factory import RotationVectorFactoryMixin


@dataclass(frozen=True, eq=False)
class RotationVector(RotationVectorFactoryMixin):
    """
    Container class for rotation vectors (axis–angle).

    The direction of the vector is the rotation axis, and its length is the rotation
    angle in radians.

    Parameters
    ----------
    value : FloatArray
        Rotation vector with shape (3,).

    Raises
    ------
    ValueError
        If ``value`` does not have shape (3,).
    """

    value: FloatArray

    def __post_init__(self) -> None:
        """Validate the vector and store a read-only copy of it."""
        if self.value.shape != (3,):
            raise ValueError(
                f"Invalid rotation vector: must be a shape (3,) array, got shape {self.value.shape}."
            )
        object.__setattr__(self, "value", to_readonly_array(self.value))

    @property
    def angle(self) -> float:
        """Return the rotation angle in radians, the length of the vector."""
        return float(np.linalg.norm(self.value))

    @property
    def rotation_matrix(self) -> FloatArray:
        """Return the 3×3 rotation matrix in the column-vector convention ``v_new = R @ v``."""
        # SciPy's Cython routines reject read-only buffers on some builds, so pass a copy.
        return Rotation.from_rotvec(np.array(self.value)).as_matrix()
