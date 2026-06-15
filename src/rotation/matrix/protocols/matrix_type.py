from __future__ import annotations

from typing import Protocol, TypeVar

from ...types import FloatArray
from .like import RotationMatrixLike

# Covariant: only used as the return type of ``RotationMatrixType.__call__`` (Protocol requirement).
_FactoryOutput = TypeVar(
    "_FactoryOutput",
    bound=RotationMatrixLike,
    covariant=True,
)


class RotationMatrixType(Protocol[_FactoryOutput]):
    """
    Class object typed as a callable factory.

    ``type[_R]`` is not enough for ``cls(*, value=...)`` because static analysis
    uses ``type.__call__``; this protocol matches concrete matrix classes.
    """

    def __call__(self, *, value: FloatArray) -> _FactoryOutput: ...
