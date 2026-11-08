from __future__ import annotations

from typing import TypeVar

from ..protocol import RotationMatrixLike

_RotationMatrix = TypeVar("_RotationMatrix", bound=RotationMatrixLike)


class RotationMatrixSpecialMixin:
    """Composition of rotation matrices with ``@``."""

    def __matmul__(self: _RotationMatrix, other: RotationMatrixLike) -> _RotationMatrix:
        """Return the composition that applies ``other`` first and then ``self``."""
        return type(self)(value=self.value @ other.value)
