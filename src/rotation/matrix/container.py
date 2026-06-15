from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ..types import FloatArray
from .mixin.axis import RotationMatrixAxisMixin
from .mixin.factory import RotationMatrixFactoryMixin
from .mixin.special import RotationMatrixSpecialMixin


@dataclass(frozen=True)
class RotationMatrix(
    RotationMatrixFactoryMixin,
    RotationMatrixAxisMixin,
    RotationMatrixSpecialMixin,
):
    """
    Container class for rotation matrix representation of rotation.

    Attributes
    ----------
    value: FloatArray
        The rotation matrix with shape (3, 3).

    Raises
    ------
    ValueError:
        If the rotation matrix is not a shape (3, 3) array.
        If the rotation matrix is not orthogonal.
        If the rotation matrix does not have determinant +1 (not a proper rotation).
    """
    value: FloatArray

    def __post_init__(self) -> None:
        """Validate that the rotation matrix is valid."""
        type(self).validate_rotation_matrix_array(self.value)

    @property
    def is_orthogonal(self) -> bool:
        """
        Check if the rotation matrix is orthogonal.

        Returns
        -------
        bool:
            True if the rotation matrix is orthogonal, False otherwise.
        """
        return np.allclose(self.value.T @ self.value, np.eye(3), atol=1e-6)

    @property
    def is_determinant_correct(self) -> bool:
        """
        Check if the rotation matrix has determinant +1 (proper rotation in SO(3)).

        Returns
        -------
        bool:
            True if the determinant is +1, False otherwise.
        """
        return np.isclose(np.linalg.det(self.value), 1.0, atol=1e-6)

    @property
    def T(self) -> FloatArray:
        """
        Return the transpose of the rotation matrix.

        Returns
        -------
        FloatArray:
            The transpose of the rotation matrix.
        """
        return self.value.T

    @property
    def inv(self) -> FloatArray:
        """
        Return the inverse of the rotation matrix.

        Returns
        -------
        FloatArray:
            The inverse of the rotation matrix.
        """
        return np.linalg.inv(self.value)

    @property
    def rotation_vector(self) -> FloatArray:
        """
        Return the rotation vector of the rotation matrix.

        Returns
        -------
        FloatArray:
            The rotation vector of the rotation matrix.
        """
        from ..vector import RotationVector
        vector: RotationVector = RotationVector.from_matrix(self.value, validate=False)
        return vector.value
