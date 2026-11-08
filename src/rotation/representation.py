from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from .types import FloatArray


class RotationRepresentation(ABC):
    """
    Operations shared by every rotation representation, single or batch.

    Each concrete class stores its own array and converts it to an internal rotation
    once, on first use, so that every operation works the same way regardless of the
    representation. Operations return a new instance of the class they are called on.
    """

    value: FloatArray
    """Stored array, whose shape and layout depend on the representation."""

    if TYPE_CHECKING:
        # Every concrete class is a dataclass whose only constructor argument is value.
        def __init__(self, *, value: FloatArray) -> None: ...

    @abstractmethod
    def _to_scipy(self) -> Rotation:
        """Return the internal rotation equivalent to the stored array."""

    @classmethod
    @abstractmethod
    def _from_scipy(cls, rotation: Rotation) -> Self:
        """Create an instance from an internal rotation."""

    @cached_property
    def _rotation(self) -> Rotation:
        return self._to_scipy()

    def inverse(self) -> Self:
        """Return the inverse rotation, which undoes this one."""
        return self._from_scipy(self._rotation.inv())

    def __pow__(self, exponent: float) -> Self:
        """
        Return the rotation about the same axis by ``exponent`` times the angle.

        The angle used is the one in [0, π], so ``r ** 0.5`` is the rotation halfway
        along the shortest path from the identity to ``r``. Integer exponents equal
        repeated composition, and ``r ** -1`` equals ``r.inverse()``.
        """
        return self._from_scipy(self._rotation ** float(exponent))

    def apply(self, vectors: npt.ArrayLike) -> FloatArray:
        """
        Rotate vectors, computing ``R @ v`` for each vector ``v``.

        Parameters
        ----------
        vectors : array_like
            One vector with shape (3,) or several with shape (M, 3). A batch of N
            rotations takes either one vector, applied to every rotation, or N
            vectors, applied pairwise.

        Returns
        -------
        FloatArray
            Rotated vectors with shape (3,) when a single rotation is applied to one
            vector, and (N, 3) or (M, 3) otherwise.

        Raises
        ------
        ValueError
            If ``vectors`` does not have shape (3,) or (M, 3), or if a batch and the
            vectors have different, non-one lengths.
        """
        array = np.asarray(vectors, dtype=np.float64)
        if array.ndim not in (1, 2) or array.shape[-1] != 3:
            raise ValueError(
                f"Invalid vectors: must be a shape (3,) or (M, 3) array, got shape {array.shape}."
            )
        return np.asarray(self._rotation.apply(array), dtype=np.float64)
