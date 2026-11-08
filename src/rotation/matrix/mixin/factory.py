from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt

from ...types import FloatArray
from ...utils import determinant
from ..protocol import RotationMatrixLike

_RotationMatrix = TypeVar("_RotationMatrix", bound=RotationMatrixLike)


class RotationMatrixFactoryMixin:
    """Factories and validation of rotation matrices."""

    @classmethod
    def unit_matrix(cls: type[_RotationMatrix]) -> _RotationMatrix:
        """Create the identity rotation matrix."""
        return cls(value=np.eye(3, dtype=np.float64))

    @classmethod
    def from_approximate_matrix_by_qr(
        cls: type[_RotationMatrix],
        value: npt.ArrayLike,
    ) -> _RotationMatrix:
        """
        Create a rotation matrix from an approximate one by QR decomposition.

        The Q factor is used, with its last column negated when needed to make the
        determinant +1.

        Parameters
        ----------
        value : array_like
            Approximate rotation matrix with shape (3, 3).

        Raises
        ------
        ValueError
            If ``value`` does not have shape (3, 3).
        """
        matrix = RotationMatrixFactoryMixin._as_square_matrix(value)
        orthogonal: FloatArray = np.linalg.qr(matrix).Q
        determinant_value = determinant(orthogonal)
        if determinant_value < 0:
            orthogonal[:, -1] *= -1
        return cls(value=orthogonal)

    @classmethod
    def from_approximate_matrix_by_svd(
        cls: type[_RotationMatrix],
        value: npt.ArrayLike,
    ) -> _RotationMatrix:
        """
        Create the rotation matrix closest to an approximate one by SVD.

        Parameters
        ----------
        value : array_like
            Approximate rotation matrix with shape (3, 3).

        Raises
        ------
        ValueError
            If ``value`` does not have shape (3, 3).
        """
        matrix = RotationMatrixFactoryMixin._as_square_matrix(value)
        decomposition = np.linalg.svd(matrix)
        left: FloatArray = decomposition.U
        right: FloatArray = decomposition.Vh
        determinant_value = determinant(left @ right)
        if determinant_value < 0:
            left[:, -1] *= -1
        return cls(value=left @ right)

    @staticmethod
    def validate_rotation_matrix_array(
        value: npt.ArrayLike,
        *,
        atol: float = 1e-6,
    ) -> None:
        """
        Check that ``value`` is a proper rotation matrix in SO(3).

        Parameters
        ----------
        value : array_like
            Matrix to check.
        atol : float, default 1e-6
            Absolute tolerance of the orthogonality and determinant checks.

        Raises
        ------
        ValueError
            If ``value`` does not have shape (3, 3), is not orthogonal, or does not
            have determinant +1.
        """
        matrix = RotationMatrixFactoryMixin._as_square_matrix(value)
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=atol):
            raise ValueError("Invalid rotation matrix: must be orthogonal.")
        determinant_value = determinant(matrix)
        if not np.isclose(determinant_value, 1.0, atol=atol):
            raise ValueError(
                f"Invalid rotation matrix: must have determinant +1, got {determinant_value}."
            )

    @staticmethod
    def _as_square_matrix(value: npt.ArrayLike) -> FloatArray:
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(
                f"Invalid rotation matrix: must be a shape (3, 3) array, got shape {matrix.shape}."
            )
        return matrix
