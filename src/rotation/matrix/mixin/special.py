from __future__ import annotations

from typing import TypeVar

from ..protocols import RotationMatrixLike, rotation_matrix_ctor

_R = TypeVar("_R", bound=RotationMatrixLike)


class RotationMatrixSpecialMixin:
    def __matmul__(self: _R, other: RotationMatrixLike) -> _R:
        """
        Multiply two rotation matrices.

        Returns
        -------
        RotationMatrix:
            The product of the two rotation matrices.
        """
        return rotation_matrix_ctor(type(self))(value=self.value @ other.value)
