from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation  # type: ignore

from ..utils import to_readonly_array
from ..types import FloatArray
from .mixin.factory import RotationVectorFactoryMixin


@dataclass(frozen=True, eq=False)
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
        """Store a read-only copy of the vector and validate it."""
        object.__setattr__(self, "value", to_readonly_array(self.value))
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
        Rotation matrix for this vector (via scipy ``Rotation.from_rotvec``).

        Returns
        -------
        FloatArray: The corresponding 3x3 rotation matrix (float64).
        """
        scipy_rotation = Rotation.from_rotvec(self.value)
        return np.asarray(scipy_rotation.as_matrix(), dtype=np.float64)
