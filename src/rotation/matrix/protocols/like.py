from __future__ import annotations

from typing import Protocol

from ...types import FloatArray


class RotationMatrixLike(Protocol):
    """Structural typing for rotation-matrix containers used by mixins."""

    value: FloatArray
