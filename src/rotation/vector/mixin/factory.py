from __future__ import annotations

from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt

from ...matrix import RotationMatrix
from ..protocol import RotationVectorLike, rotation_vector_ctor

_R = TypeVar("_R", bound=RotationVectorLike)


class RotationVectorFactoryMixin:
    @classmethod
    def from_matrix(
        cls: type[_R],
        rotation_matrix: npt.ArrayLike,
        *,
        validate: bool = True,
    ) -> _R:
        """
        Create a RotationVector from a 3×3 rotation matrix array.

        By default the array is checked to lie in SO(3) (orthogonal, determinant +1),
        matching :class:`RotationMatrix`, so arbitrary ``(3, 3)`` arrays cannot slip
        through silently. Pass ``validate=False`` only when the matrix is already
        known to be a valid rotation (for example from :class:`RotationMatrix`).

        Parameters
        ----------
        rotation_matrix : array_like
            The source rotation matrix with shape (3, 3).
        validate : bool, default True
            If True, run :meth:`RotationMatrix.validate_rotation_matrix_array`.

        Returns
        -------
        RotationVector: The resulting rotation vector.

        Raises
        ------
        ValueError:
            If ``rotation_matrix`` is not shape (3, 3), or if ``validate`` is True
            and the matrix is not a proper rotation.
        """
        R = np.asarray(rotation_matrix, dtype=np.float64)
        if validate:
            RotationMatrix.validate_rotation_matrix_array(R)
        elif R.shape != (3, 3):
            raise ValueError(
                f"from_matrix expects shape (3, 3), got {R.shape}."
            )
        cos_theta = (np.trace(R) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        if np.isclose(theta, 0.0):
            return cast(_R, cls.zero_vector())

        axis_unnormalized = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ],
            dtype=np.float64,
        )

        norm = float(np.linalg.norm(axis_unnormalized))
        if np.isclose(norm, 0.0):
            vals, vecs = np.linalg.eigh(R)
            axis = cast(
                npt.NDArray[np.float64],
                vecs[:, np.isclose(vals, 1.0)][:, 0],
            )
        else:
            axis = axis_unnormalized / norm

        return rotation_vector_ctor(cls)(
            value=axis * np.float64(theta),
        )

    @classmethod
    def zero_vector(cls: type[_R]) -> _R:
        """
        Create an identity rotation (zero vector).

        Returns
        -------
        RotationVector: The zero rotation vector.
        """
        return rotation_vector_ctor(cls)(value=np.zeros(3, dtype=np.float64))

    @classmethod
    def from_axis_angle(
        cls: type[_R],
        axis: npt.ArrayLike,
        angle: float,
    ) -> _R:
        """
        Create a rotation vector from a specific axis and angle.

        Parameters
        ----------
        axis : array_like
            The rotation axis (will be normalized).
        angle : float
            The rotation angle in radians.
        """
        a = np.asarray(axis, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(a))
        if np.isclose(norm, 0.0):
            return cast(_R, cls.zero_vector())

        unit_axis = a / norm
        return rotation_vector_ctor(cls)(value=unit_axis * np.float64(angle))
