from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Self

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

    def _with_rotation(self, rotation: Rotation) -> Self:
        """Create an instance like this one, such as with the same format, for ``rotation``."""
        return self._from_scipy(rotation)

    @property
    def rotation_matrix(self) -> FloatArray:
        """
        Return the rotation as a matrix in the column-vector convention ``v_new = R @ v``.

        The shape is (3, 3) for a single rotation and (N, 3, 3) for a batch.
        """
        return np.asarray(self._rotation.as_matrix(), dtype=np.float64)

    @property
    def rotation_vector(self) -> FloatArray:
        """
        Return the rotation as an axis scaled by the angle in radians.

        The shape is (3,) for a single rotation and (N, 3) for a batch. Rotation
        vectors return their stored value; other representations return the vector
        whose length lies in [0, π].
        """
        return np.asarray(self._rotation.as_rotvec(), dtype=np.float64)

    @property
    def wxyz(self) -> FloatArray:
        """
        Return the rotation as unit quaternion components in scalar-first order.

        The shape is (4,) for a single rotation and (N, 4) for a batch. Quaternions
        return their stored value; other representations return either sign, since
        ``q`` and ``-q`` are the same rotation.
        """
        return np.asarray(self._rotation.as_quat(scalar_first=True), dtype=np.float64)

    @property
    def xyzw(self) -> FloatArray:
        """
        Return the rotation as unit quaternion components in scalar-last order.

        The shape is (4,) for a single rotation and (N, 4) for a batch.
        """
        return np.roll(self.wxyz, -1, axis=-1)

    @property
    def inverse(self) -> Self:
        """Return the inverse rotation, which undoes this one, in the same representation."""
        return self._with_rotation(self._rotation.inv())

    def __pow__(self, exponent: float) -> Self:
        """
        Return the rotation about the same axis by ``exponent`` times the angle.

        The angle used is the one in [0, π], so ``r ** 0.5`` is the rotation halfway
        along the shortest path from the identity to ``r``. Integer exponents equal
        repeated composition, and ``r ** -1`` equals ``r.inverse``.
        """
        return self._with_rotation(self._rotation ** float(exponent))

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

    @staticmethod
    def _random_scipy(
        generator: np.random.Generator, shape: tuple[int, ...]
    ) -> Rotation:
        # Normalized 4D Gaussian samples are uniform on the unit quaternions, hence on SO(3).
        components = generator.standard_normal((*shape, 4))
        components /= np.linalg.norm(components, axis=-1, keepdims=True)
        return Rotation.from_quat(components, scalar_first=True)

    @staticmethod
    def _as_weights(weights: npt.ArrayLike | None, count: int) -> FloatArray | None:
        if weights is None:
            return None
        array = np.asarray(weights, dtype=np.float64)
        if array.shape != (count,):
            raise ValueError(
                f"Invalid weights: must be a shape ({count},) array, got shape {array.shape}."
            )
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError("Invalid weights: must be finite and non-negative.")
        if not np.any(array > 0.0):
            raise ValueError("Invalid weights: must have at least one positive weight.")
        return array
