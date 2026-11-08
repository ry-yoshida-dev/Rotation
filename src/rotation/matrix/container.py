from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..types import FloatArray
from ..utils import determinant, to_readonly_array
from .mixin.axis import RotationMatrixAxisMixin
from .mixin.factory import RotationMatrixFactoryMixin
from .mixin.special import RotationMatrixSpecialMixin


@dataclass(frozen=True, eq=False)
class RotationMatrix(
    RotationMatrixFactoryMixin,
    RotationMatrixAxisMixin,
    RotationMatrixSpecialMixin,
):
    """
    Container class for rotation matrices in SO(3).

    Matrices follow the column-vector convention ``v_new = R @ v``.

    Parameters
    ----------
    value : FloatArray
        Rotation matrix with shape (3, 3).

    Raises
    ------
    ValueError
        If ``value`` does not have shape (3, 3), is not orthogonal, or does not have
        determinant +1.
    """

    value: FloatArray

    def __post_init__(self) -> None:
        """Store a read-only copy of the matrix and validate it."""
        type(self).validate_rotation_matrix_array(self.value)
        object.__setattr__(self, "value", to_readonly_array(self.value))

    @property
    def is_orthogonal(self) -> bool:
        """Return whether the matrix is orthogonal within an absolute tolerance of 1e-6."""
        return np.allclose(self.value.T @ self.value, np.eye(3), atol=1e-6)

    @property
    def is_determinant_correct(self) -> bool:
        """Return whether the determinant is +1 within an absolute tolerance of 1e-6."""
        determinant_value = determinant(self.value)
        return bool(np.isclose(determinant_value, 1.0, atol=1e-6))

    @property
    def T(self) -> FloatArray:
        """Return the transpose of the matrix."""
        return self.value.T

    @property
    def inverse(self) -> FloatArray:
        """Return the inverse of the matrix, which equals its transpose."""
        return self.value.T

    @property
    def rotation_vector(self) -> FloatArray:
        """Return the equivalent rotation vector, whose angle lies in [0, π]."""
        return Rotation.from_matrix(self.value).as_rotvec()
