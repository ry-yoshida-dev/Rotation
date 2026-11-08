from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ..batch import BatchRotation
from ..types import FloatArray
from ..utils import to_readonly_array
from .single import RotationVector


@dataclass(frozen=True, eq=False)
class RotationVectors(BatchRotation[RotationVector]):
    """
    N rotations stored as rotation vectors (axis–angle).

    Each row's direction is its rotation axis and its length is its rotation angle in
    radians. Conversions from other representations return lengths in [0, π].

    Parameters
    ----------
    value : FloatArray
        Rotation vectors with shape (N, 3).

    Raises
    ------
    ValueError
        If ``value`` does not have shape (N, 3) or contains non-finite components.
    """

    value: FloatArray

    def __post_init__(self) -> None:
        """Validate the vectors and store a read-only float64 copy of them."""
        vectors = np.asarray(self.value, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError(
                f"Invalid rotation vectors: must be a shape (N, 3) array, got shape {vectors.shape}."
            )
        if not np.all(np.isfinite(vectors)):
            raise ValueError("Invalid rotation vectors: must be finite.")
        object.__setattr__(self, "value", to_readonly_array(vectors))

    @classmethod
    def from_axes_angles(cls, axes: npt.ArrayLike, angles: npt.ArrayLike) -> Self:
        """
        Create rotation vectors from rotation axes and angles.

        Parameters
        ----------
        axes : array_like
            Rotation axes with shape (N, 3). They do not need to be normalized.
        angles : array_like
            Rotation angles in radians with shape (N,).

        Raises
        ------
        ValueError
            If ``axes`` does not have shape (N, 3), ``angles`` does not have shape
            (N,), or an axis has zero length.
        """
        axes_array = np.asarray(axes, dtype=np.float64)
        angles_array = np.asarray(angles, dtype=np.float64)
        if axes_array.ndim != 2 or axes_array.shape[1] != 3:
            raise ValueError(
                f"Invalid rotation axes: must be a shape (N, 3) array, got shape {axes_array.shape}."
            )
        if angles_array.shape != (len(axes_array),):
            raise ValueError(
                f"Invalid angles: must be a shape ({len(axes_array)},) array, got shape {angles_array.shape}."
            )
        norms = np.linalg.norm(axes_array, axis=1)
        if np.any(np.isclose(norms, 0.0)):
            raise ValueError("Invalid rotation axes: must have non-zero length.")
        return cls(value=axes_array * (angles_array / norms)[:, np.newaxis])

    @property
    def lengths(self) -> FloatArray:
        """
        Return the length of each stored vector in radians, with shape (N,).

        Unlike ``angles``, they are not reduced to [0, π].
        """
        return np.linalg.norm(self.value, axis=1)

    @property
    def axes(self) -> FloatArray:
        """
        Return the unit rotation axes with shape (N, 3).

        Raises
        ------
        ValueError
            If a vector has zero length, so that its axis is undefined.
        """
        lengths = self.lengths
        if np.any(np.isclose(lengths, 0.0)):
            raise ValueError("The axis of the identity rotation is undefined.")
        return self.value / lengths[:, np.newaxis]

    @classmethod
    def _single_type(cls) -> type[RotationVector]:
        return RotationVector

    def _to_scipy(self) -> Rotation:
        return Rotation.from_rotvec(np.array(self.value).reshape(-1, 3))

    @classmethod
    def _from_scipy(cls, rotation: Rotation) -> Self:
        return cls(value=rotation.as_rotvec().reshape(-1, 3))
