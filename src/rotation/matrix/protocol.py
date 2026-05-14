from __future__ import annotations

from typing import Protocol, TypeVar, cast

import numpy as np

_R_co = TypeVar("_R_co", bound="RotationMatrixLike", covariant=True)
_R_ctor = TypeVar("_R_ctor", bound="RotationMatrixLike")


class RotationMatrixLike(Protocol):
    """Structural typing for rotation-matrix containers used by mixins."""

    value: np.ndarray


class RotationMatrixType(Protocol[_R_co]):
    """
    Class object typed as a callable factory.

    ``type[_R]`` is not enough for ``cls(*, value=...)`` because static analysis
    uses ``type.__call__``; this protocol matches concrete matrix classes.
    """

    def __call__(self, *, value: np.ndarray) -> _R_co: ...


def rotation_matrix_ctor(cls: type[_R_ctor]) -> RotationMatrixType[_R_ctor]:
    """Narrow ``type[_R]`` / ``type(self)`` for ``(*, value: ndarray) -> _R`` (see ``RotationMatrixType``)."""
    return cast(RotationMatrixType[_R_ctor], cls)
