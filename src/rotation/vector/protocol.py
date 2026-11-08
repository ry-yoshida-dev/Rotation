from __future__ import annotations

from typing import Protocol

from ..types import FloatArray


class RotationVectorLike(Protocol):
    """
    Structural type of rotation vector containers.

    Any class with a keyword ``value`` constructor and a ``value`` attribute matches,
    so code can build the concrete subclass from ``cls`` or ``type(self)``.
    """

    def __init__(self, *, value: FloatArray) -> None: ...

    @property
    def value(self) -> FloatArray:
        """Return the rotation vector with shape (3,)."""
        ...
