from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ..matrix import RotationMatrix
from ..single import SingleRotation
from ..types import FloatArray
from ..utils import to_readonly_array
from .skew_symmetric import SkewSymmetricMatrix


@dataclass(frozen=True, eq=False)
class RotationVector(SingleRotation):
    """
    Rotation stored as a rotation vector (axis–angle).

    The direction of the vector is the rotation axis, and its length is the rotation
    angle in radians, counter-clockwise when the axis points toward the viewer. Any
    length is accepted; conversions from other representations return the vector
    whose length lies in [0, π].

    Parameters
    ----------
    value : FloatArray
        Rotation vector with shape (3,).

    Raises
    ------
    ValueError
        If ``value`` does not have shape (3,) or contains non-finite components.
    """

    value: FloatArray

    def __post_init__(self) -> None:
        """Validate the vector and store a read-only float64 copy of it."""
        vector = np.asarray(self.value, dtype=np.float64)
        if vector.shape != (3,):
            raise ValueError(
                f"Invalid rotation vector: must be a shape (3,) array, got shape {vector.shape}."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("Invalid rotation vector: must be finite.")
        object.__setattr__(self, "value", to_readonly_array(vector))

    @classmethod
    def from_matrix(cls, rotation_matrix: npt.ArrayLike) -> Self:
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
        return cls.from_rotation(RotationMatrix(value=np.asarray(rotation_matrix)))

    @classmethod
    def zero_vector(cls) -> Self:
        """Create the identity rotation, the zero vector."""
        return cls.identity()

    @classmethod
    def from_axis_angle(cls, axis: npt.ArrayLike, angle: float) -> Self:
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

    @classmethod
    def from_skew_matrix(cls, matrix: npt.ArrayLike) -> Self:
        """
        Create a rotation vector from its skew-symmetric matrix, the vee map.

        Parameters
        ----------
        matrix : array_like
            Skew-symmetric matrix ``[ω]×`` with shape (3, 3).

        Raises
        ------
        ValueError
            If ``matrix`` does not have shape (3, 3) or is not skew-symmetric.
        """
        array = np.asarray(matrix, dtype=np.float64)
        if array.shape != (3, 3):
            raise ValueError(
                f"Invalid matrix: must be a shape (3, 3) array, got shape {array.shape}."
            )
        return cls(value=SkewSymmetricMatrix.to_vectors(array))

    @property
    def skew_matrix(self) -> FloatArray:
        """
        Return the skew-symmetric matrix ``[ω]×`` with shape (3, 3), the hat map.

        It is the Lie algebra element of the rotation, and ``[ω]× @ v = ω × v``.
        """
        return SkewSymmetricMatrix.from_vectors(self.value)

    @property
    def angle(self) -> float:
        """
        Return the rotation angle in radians, the length of the stored vector.

        Unlike other representations, it is not reduced to [0, π].
        """
        return float(np.linalg.norm(self.value))

    @property
    def axis(self) -> FloatArray:
        """
        Return the unit rotation axis with shape (3,).

        Raises
        ------
        ValueError
            If the vector has zero length, so that the axis is undefined.
        """
        length = self.angle
        if np.isclose(length, 0.0):
            raise ValueError("The axis of the identity rotation is undefined.")
        return self.value / length

    @property
    def rotation_vector(self) -> FloatArray:
        """Return the stored vector with shape (3,), whose length is not reduced to [0, π]."""
        return self.value

    def _value_from_scipy(self, rotation: Rotation) -> FloatArray:
        return rotation.as_rotvec()

    def _with_value(self, value: FloatArray) -> Self:
        return replace(self, value=value)

    def _to_scipy(self) -> Rotation:
        return Rotation.from_rotvec(np.array(self.value))

    @classmethod
    def _from_scipy(cls, rotation: Rotation) -> Self:
        return cls(value=rotation.as_rotvec())
