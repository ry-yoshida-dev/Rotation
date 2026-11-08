from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ..single import SingleRotation
from ..types import FloatArray
from ..utils import to_readonly_array
from .special_orthogonal import SpecialOrthogonalMatrix


@dataclass(frozen=True, eq=False)
class RotationMatrix(SingleRotation):
    """
    Rotation stored as a 3×3 matrix in SO(3).

    Matrices follow the column-vector convention ``v_new = R @ v``, so each column is
    the image of the corresponding basis vector.

    Parameters
    ----------
    value : FloatArray
        Rotation matrix with shape (3, 3).

    Raises
    ------
    ValueError
        If ``value`` does not have shape (3, 3), is not orthogonal, or does not have
        determinant +1, within an absolute tolerance of 1e-6.
    """

    value: FloatArray

    def __post_init__(self) -> None:
        """Validate the matrix and store a read-only float64 copy of it."""
        matrix = RotationMatrix._as_square_matrix(self.value)
        SpecialOrthogonalMatrix.validate(matrix)
        object.__setattr__(self, "value", to_readonly_array(matrix))

    @classmethod
    def from_approximate_matrix_by_svd(cls, value: npt.ArrayLike) -> Self:
        """
        Create the rotation matrix closest to an approximate one in the Frobenius norm.

        Parameters
        ----------
        value : array_like
            Approximate rotation matrix with shape (3, 3).

        Raises
        ------
        ValueError
            If ``value`` does not have shape (3, 3).
        """
        matrix = RotationMatrix._as_square_matrix(value)
        return cls(value=SpecialOrthogonalMatrix.project_by_svd(matrix))

    @classmethod
    def from_approximate_matrix_by_qr(cls, value: npt.ArrayLike) -> Self:
        """
        Create a rotation matrix from an approximate one by QR decomposition.

        The result keeps the direction of the first column. It is cheaper than
        ``from_approximate_matrix_by_svd`` but not the closest rotation in general.

        Parameters
        ----------
        value : array_like
            Approximate rotation matrix with shape (3, 3).

        Raises
        ------
        ValueError
            If ``value`` does not have shape (3, 3).
        """
        matrix = RotationMatrix._as_square_matrix(value)
        return cls(value=SpecialOrthogonalMatrix.project_by_qr(matrix))

    @property
    def x_axis(self) -> FloatArray:
        """Return the first column, the rotated x-axis, with shape (3,)."""
        return self.value[:, 0]

    @property
    def y_axis(self) -> FloatArray:
        """Return the second column, the rotated y-axis, with shape (3,)."""
        return self.value[:, 1]

    @property
    def z_axis(self) -> FloatArray:
        """Return the third column, the rotated z-axis, with shape (3,)."""
        return self.value[:, 2]

    def _to_scipy(self) -> Rotation:
        return Rotation.from_matrix(np.array(self.value))

    @classmethod
    def _from_scipy(cls, rotation: Rotation) -> Self:
        return cls(value=rotation.as_matrix())

    @staticmethod
    def _as_square_matrix(value: npt.ArrayLike) -> FloatArray:
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(
                f"Invalid rotation matrix: must be a shape (3, 3) array, got shape {matrix.shape}."
            )
        return matrix
