from __future__ import annotations

from typing import Protocol, TypeVar, cast

import numpy as np

# Covariant: only used as the return type of ``RotationVectorType.__call__`` (Protocol requirement).
_FactoryOutput = TypeVar(
    "_FactoryOutput",
    bound="RotationVectorLike",
    covariant=True,
)
# The concrete ``type[...]`` passed to ``rotation_vector_ctor`` (same idea as ``_R`` in mixins).
_Subclass = TypeVar("_Subclass", bound="RotationVectorLike")


class RotationVectorLike(Protocol):
    """Structural typing for rotation-vector containers used by mixins."""

    value: np.ndarray

    @classmethod
    def zero_vector(cls) -> RotationVectorLike:
        ...


class RotationVectorType(Protocol[_FactoryOutput]):
    """
    Class object typed as a callable factory.

    Same rationale as :class:`RotationMatrixType` in ``matrix.protocol``.
    """

    def __call__(self, *, value: np.ndarray) -> _FactoryOutput: ...


def rotation_vector_ctor(cls: type[_Subclass]) -> RotationVectorType[_Subclass]:
    """Narrow ``type[_R]`` / ``type(self)`` for ``(*, value: ndarray) -> _R`` calls."""
    return cast(RotationVectorType[_Subclass], cls)
