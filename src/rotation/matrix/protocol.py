from __future__ import annotations

from typing import Protocol

from ..types import FloatArray


class RotationMatrixLike(Protocol):
    """
    Structural type of rotation matrix containers.

    Declaring the keyword constructor lets the mixins build the concrete subclass from
    ``cls`` or ``type(self)``.
    """

    def __init__(self, *, value: FloatArray) -> None: ...

    @property
    def value(self) -> FloatArray:
        """Return the rotation matrix with shape (3, 3)."""
        ...
