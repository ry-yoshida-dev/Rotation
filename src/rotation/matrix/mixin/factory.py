from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt

from ...types import FloatArray
from ..protocols import RotationMatrixLike, rotation_matrix_ctor

_R = TypeVar("_R", bound=RotationMatrixLike)


class RotationMatrixFactoryMixin:
    @classmethod
    def unit_matrix(cls: type[_R]) -> _R:
        """
        Create the identity rotation matrix.

        Returns
        -------
        RotationMatrix:
            The unit rotation matrix.
        """
        return rotation_matrix_ctor(cls)(value=np.eye(3, dtype=np.float64))

    @classmethod
    def from_approximate_matrix_by_qr(
        cls: type[_R],
        value: npt.ArrayLike,
    ) -> _R:
        """
        Create a RotationMatrix object from an approximate rotation matrix by QR decomposition.

        Parameters
        ----------
        value: array_like
            The approximate rotation matrix.

        Returns
        -------
        RotationMatrix:
            The rotation matrix from the approximate matrix.
        """
        arr = np.asarray(value, dtype=np.float64)
        q, _ = np.linalg.qr(arr)
        det = np.linalg.det(q)

        if det < 0:
            q = q.copy()
            q[:, -1] *= -1
        return rotation_matrix_ctor(cls)(value=q)

    @classmethod
    def from_approximate_matrix_with_SVD(
        cls: type[_R],
        value: npt.ArrayLike,
    ) -> _R:
        """
        Create a RotationMatrix object from an approximate rotation matrix by SVD decomposition.

        Parameters
        ----------
        value: array_like
            The approximate rotation matrix.

        Returns
        -------
        RotationMatrix:
            The rotation matrix from the approximate matrix.
        """
        arr = np.asarray(value, dtype=np.float64)
        u, _, vh = np.linalg.svd(arr)
        rotation_matrix: FloatArray = u @ vh

        if np.linalg.det(rotation_matrix) < 0:
            u = u.copy()
            u[:, -1] *= -1
            rotation_matrix = u @ vh

        return rotation_matrix_ctor(cls)(value=rotation_matrix)

    @staticmethod
    def validate_rotation_matrix_array(
        value: npt.ArrayLike,
        *,
        atol: float = 1e-6,
    ) -> None:
        """
        Raise ``ValueError`` unless ``value`` is a proper 3×3 rotation matrix in SO(3).

        Use this when accepting a raw ``FloatArray`` so invalid matrices cannot slip through
        without constructing :class:`RotationMatrix`.
        """
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(
                f"Invalid rotation matrix: must be a shape (3, 3) array, got shape {arr.shape}."
            )
        if not np.allclose(arr.T @ arr, np.eye(3), atol=atol):
            raise ValueError("Invalid rotation matrix: must be orthogonal")
        if not np.isclose(np.linalg.det(arr), 1.0, atol=atol):
            raise ValueError("Invalid rotation matrix: must have correct determinant")
