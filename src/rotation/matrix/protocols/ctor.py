from __future__ import annotations

from typing import TypeVar, cast

from .like import RotationMatrixLike
from .matrix_type import RotationMatrixType

# The concrete ``type[...]`` passed to ``rotation_matrix_ctor`` (same idea as ``_R`` in mixins).
_Subclass = TypeVar("_Subclass", bound=RotationMatrixLike)


def rotation_matrix_ctor(cls: type[_Subclass]) -> RotationMatrixType[_Subclass]:
    """Narrow ``type[_R]`` / ``type(self)`` for ``(*, value: ndarray) -> _R`` (see ``RotationMatrixType``)."""
    return cast(RotationMatrixType[_Subclass], cls)
