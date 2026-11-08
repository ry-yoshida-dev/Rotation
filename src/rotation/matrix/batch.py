from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ..batch import BatchRotation
from ..types import FloatArray
from ..utils import to_readonly_array
from .single import RotationMatrix
from .special_orthogonal import SpecialOrthogonalMatrix


@dataclass(frozen=True, eq=False)
class RotationMatrices(BatchRotation[RotationMatrix]):
    """
    N rotations stored as 3×3 matrices in SO(3).

    Matrices follow the column-vector convention ``v_new = R @ v``.

    Parameters
    ----------
    value : FloatArray
        Rotation matrices with shape (N, 3, 3).

    Raises
    ------
    ValueError
        If ``value`` does not have shape (N, 3, 3), or if a matrix is not orthogonal
        or does not have determinant +1, within an absolute tolerance of 1e-6.
    """

    value: FloatArray

    def __post_init__(self) -> None:
        """Validate the matrices and store a read-only float64 copy of them."""
        matrices = RotationMatrices._as_matrices(self.value)
        SpecialOrthogonalMatrix.validate(matrices)
        object.__setattr__(self, "value", to_readonly_array(matrices))

    @classmethod
    def from_approximate_matrices_by_svd(cls, value: npt.ArrayLike) -> Self:
        """
        Create the rotation matrix closest to each approximate one in the Frobenius norm.

        Parameters
        ----------
        value : array_like
            Approximate rotation matrices with shape (N, 3, 3).

        Raises
        ------
        ValueError
            If ``value`` does not have shape (N, 3, 3).
        """
        matrices = RotationMatrices._as_matrices(value)
        return cls(value=SpecialOrthogonalMatrix.project_by_svd(matrices))

    @classmethod
    def from_approximate_matrices_by_qr(cls, value: npt.ArrayLike) -> Self:
        """
        Create rotation matrices from approximate ones by QR decomposition.

        Each result keeps the direction of its first column.

        Parameters
        ----------
        value : array_like
            Approximate rotation matrices with shape (N, 3, 3).

        Raises
        ------
        ValueError
            If ``value`` does not have shape (N, 3, 3).
        """
        matrices = RotationMatrices._as_matrices(value)
        return cls(value=SpecialOrthogonalMatrix.project_by_qr(matrices))

    @property
    def x_axes(self) -> FloatArray:
        """Return the first columns, the rotated x-axes, with shape (N, 3)."""
        return self.value[:, :, 0]

    @property
    def y_axes(self) -> FloatArray:
        """Return the second columns, the rotated y-axes, with shape (N, 3)."""
        return self.value[:, :, 1]

    @property
    def z_axes(self) -> FloatArray:
        """Return the third columns, the rotated z-axes, with shape (N, 3)."""
        return self.value[:, :, 2]

    @classmethod
    def _single_type(cls) -> type[RotationMatrix]:
        return RotationMatrix

    def _to_scipy(self) -> Rotation:
        return Rotation.from_matrix(np.array(self.value).reshape(-1, 3, 3))

    @classmethod
    def _from_scipy(cls, rotation: Rotation) -> Self:
        return cls(value=rotation.as_matrix().reshape(-1, 3, 3))

    @staticmethod
    def _as_matrices(value: npt.ArrayLike) -> FloatArray:
        matrices = np.asarray(value, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
            raise ValueError(
                f"Invalid rotation matrices: must be a shape (N, 3, 3) array, got shape {matrices.shape}."
            )
        return matrices
