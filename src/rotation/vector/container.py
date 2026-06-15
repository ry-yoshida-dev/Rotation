from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

from ..types import FloatArray
from .mixin.factory import RotationVectorFactoryMixin


@dataclass(frozen=True)
class RotationVector(RotationVectorFactoryMixin):
    """
    Container class for rotation vector (axis-angle) representation.

    The vector direction represents the axis of rotation, and its
    magnitude represents the rotation angle in radians.

    Attributes
    ----------
    value : FloatArray
        The rotation vector with shape (3,).

    Raises
    ------
    ValueError:
        If the input array is not a shape (3,) array.
    """
    value: FloatArray

    def __post_init__(self) -> None:
        """Validate the rotation vector."""
        self._is_valid_rotation_vector()

    def _is_valid_rotation_vector(self) -> None:
        """Check if the vector has the correct shape."""
        if self.value.shape != (3,):
            raise ValueError(
                f"Invalid rotation vector: must be a shape (3,) array, got {self.value.shape}."
            )

    @property
    def angle(self) -> float:
        """
        Return the rotation angle θ in radians.

        Returns
        -------
        float: The rotation angle (norm of the vector).
        """
        return float(np.linalg.norm(self.value))

    @property
    def rotation_matrix(self) -> FloatArray:
        """
        Rotation matrix for this vector (Rodrigues' formula via OpenCV cv2.Rodrigues).

        Returns
        -------
        FloatArray: The corresponding 3x3 rotation matrix (float64).
        """
        rvec = np.asarray(self.value, dtype=np.float64).reshape(3, 1)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        return np.asarray(rotation_matrix, dtype=np.float64)
