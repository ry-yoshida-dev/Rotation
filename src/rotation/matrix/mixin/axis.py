from __future__ import annotations

import numpy as np

from ..protocol import RotationMatrixLike


class RotationMatrixAxisMixin:
    @property
    def x_axis(self: RotationMatrixLike) -> np.ndarray:
        """
        Return the x-axis of the rotation matrix.

        Returns
        -------
        np.ndarray:
            The x-axis of the rotation matrix with shape (3,).
        """
        return self.value[:, 0]

    @property
    def y_axis(self: RotationMatrixLike) -> np.ndarray:
        """
        Return the y-axis of the rotation matrix.

        Returns
        -------
        np.ndarray:
            The y-axis of the rotation matrix with shape (3,).
        """
        return self.value[:, 1]

    @property
    def z_axis(self: RotationMatrixLike) -> np.ndarray:
        """
        Return the z-axis of the rotation matrix.

        Returns
        -------
        np.ndarray:
            The z-axis of the rotation matrix with shape (3,).
        """
        return self.value[:, 2]
