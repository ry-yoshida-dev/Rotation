from __future__ import annotations

from typing import Protocol, TypeVar, cast

import numpy as np

# Covariant: only used as the return type of ``RotationMatrixType.__call__`` (Protocol requirement).
_FactoryOutput = TypeVar(
    "_FactoryOutput",
    bound="RotationMatrixLike",
    covariant=True,
)
# The concrete ``type[...]`` passed to ``rotation_matrix_ctor`` (same idea as ``_R`` in mixins).
_Subclass = TypeVar("_Subclass", bound="RotationMatrixLike")


class RotationMatrixLike(Protocol):
    """Structural typing for rotation-matrix containers used by mixins."""

    value: np.ndarray


class RotationMatrixType(Protocol[_FactoryOutput]):
    """
    Class object typed as a callable factory.

    ``type[_R]`` is not enough for ``cls(*, value=...)`` because static analysis
    uses ``type.__call__``; this protocol matches concrete matrix classes.
    """

    def __call__(self, *, value: np.ndarray) -> _FactoryOutput: ...


def rotation_matrix_ctor(cls: type[_Subclass]) -> RotationMatrixType[_Subclass]:
    """Narrow ``type[_R]`` / ``type(self)`` for ``(*, value: ndarray) -> _R`` (see ``RotationMatrixType``)."""
    return cast(RotationMatrixType[_Subclass], cls)
