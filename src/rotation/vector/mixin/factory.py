from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ...matrix import RotationMatrix
from ..protocol import RotationVectorLike

_RotationVector = TypeVar("_RotationVector", bound=RotationVectorLike)


class RotationVectorFactoryMixin:
    """Factories of rotation vectors."""

    @classmethod
    def from_matrix(
        cls: type[_RotationVector],
        rotation_matrix: npt.ArrayLike,
    ) -> _RotationVector:
        """
        Create a rotation vector from a rotation matrix.

        Parameters
        ----------
        rotation_matrix : array_like
            Rotation matrix with shape (3, 3) in SO(3).

        Returns
        -------
        RotationVector
            Rotation vector whose angle lies in [0, π].

        Raises
        ------
        ValueError
            If ``rotation_matrix`` is not a proper rotation matrix with shape (3, 3).
        """
        matrix = np.asarray(rotation_matrix, dtype=np.float64)
        RotationMatrix.validate_rotation_matrix_array(matrix)
        return cls(value=Rotation.from_matrix(matrix).as_rotvec())

    @classmethod
    def zero_vector(cls: type[_RotationVector]) -> _RotationVector:
        """Create the identity rotation, the zero vector."""
        return cls(value=np.zeros(3, dtype=np.float64))

    @classmethod
    def from_axis_angle(
        cls: type[_RotationVector],
        axis: npt.ArrayLike,
        angle: float,
    ) -> _RotationVector:
        """
        Create a rotation vector from a rotation axis and an angle.

        Parameters
        ----------
        axis : array_like
            Rotation axis with shape (3,). It does not need to be normalized.
        angle : float
            Rotation angle in radians.

        Raises
        ------
        ValueError
            If ``axis`` does not have shape (3,) or has zero length.
        """
        axis_array = np.asarray(axis, dtype=np.float64)
        if axis_array.shape != (3,):
            raise ValueError(
                f"Invalid rotation axis: must be a shape (3,) array, got shape {axis_array.shape}."
            )
        norm = float(np.linalg.norm(axis_array))
        if np.isclose(norm, 0.0):
            raise ValueError("Invalid rotation axis: must have non-zero length.")
        return cls(value=axis_array * (angle / norm))
