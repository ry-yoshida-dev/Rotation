from __future__ import annotations

from ...types import FloatArray
from ..protocols import RotationMatrixLike


class RotationMatrixAxisMixin:
    @property
    def x_axis(self: RotationMatrixLike) -> FloatArray:
        """
        Return the x-axis of the rotation matrix.

        Returns
        -------
        FloatArray:
            The x-axis of the rotation matrix with shape (3,).
        """
        return self.value[:, 0]

    @property
    def y_axis(self: RotationMatrixLike) -> FloatArray:
        """
        Return the y-axis of the rotation matrix.

        Returns
        -------
        FloatArray:
            The y-axis of the rotation matrix with shape (3,).
        """
        return self.value[:, 1]

    @property
    def z_axis(self: RotationMatrixLike) -> FloatArray:
        """
        Return the z-axis of the rotation matrix.

        Returns
        -------
        FloatArray:
            The z-axis of the rotation matrix with shape (3,).
        """
        return self.value[:, 2]
