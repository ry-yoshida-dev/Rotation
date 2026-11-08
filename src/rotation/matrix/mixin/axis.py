from __future__ import annotations

from ...types import FloatArray
from ..protocol import RotationMatrixLike


class RotationMatrixAxisMixin:
    """Column axes of a rotation matrix."""

    @property
    def x_axis(self: RotationMatrixLike) -> FloatArray:
        """Return the first column, the rotated x-axis, with shape (3,)."""
        return self.value[:, 0]

    @property
    def y_axis(self: RotationMatrixLike) -> FloatArray:
        """Return the second column, the rotated y-axis, with shape (3,)."""
        return self.value[:, 1]

    @property
    def z_axis(self: RotationMatrixLike) -> FloatArray:
        """Return the third column, the rotated z-axis, with shape (3,)."""
        return self.value[:, 2]
