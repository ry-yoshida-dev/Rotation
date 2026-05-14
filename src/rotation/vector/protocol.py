from __future__ import annotations

from typing import Protocol, TypeVar, cast

import numpy as np

_R_co = TypeVar("_R_co", bound="RotationVectorLike", covariant=True)
_R_ctor = TypeVar("_R_ctor", bound="RotationVectorLike")


class RotationVectorLike(Protocol):
    """Structural typing for rotation-vector containers used by mixins."""

    value: np.ndarray


class RotationVectorType(Protocol[_R_co]):
    """
    Class object typed as a callable factory.

    Same rationale as :class:`RotationMatrixType` in ``matrix.protocol``.
    """

    def __call__(self, *, value: np.ndarray) -> _R_co: ...


def rotation_vector_ctor(cls: type[_R_ctor]) -> RotationVectorType[_R_ctor]:
    """Narrow ``type[_R]`` / ``type(self)`` for ``(*, value: ndarray) -> _R`` calls."""
    return cast(RotationVectorType[_R_ctor], cls)
