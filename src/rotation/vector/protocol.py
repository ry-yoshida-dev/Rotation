from __future__ import annotations

from typing import Protocol

from ..types import FloatArray


class RotationVectorLike(Protocol):
    """
    Structural type of rotation vector containers.

    Declaring the keyword constructor lets the mixins build the concrete subclass from
    ``cls``.
    """

    def __init__(self, *, value: FloatArray) -> None: ...

    @property
    def value(self) -> FloatArray:
        """Return the rotation vector with shape (3,)."""
        ...
