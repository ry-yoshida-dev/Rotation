from __future__ import annotations

from dataclasses import dataclass
from typing import Self, cast

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ..single import SingleRotation
from ..types import FloatArray
from ..utils import to_readonly_array
from .format import QuaternionFormat


@dataclass(frozen=True, eq=False)
class Quaternion(SingleRotation):
    """
    Rotation stored as a unit quaternion ``w + xi + yj + zk``.

    ``value`` keeps the component order given by ``format``. Operations on a
    quaternion, such as composition, inverse, and powers, return quaternions in the
    same format. Factories from other representations take a ``format`` argument.

    ``q`` and ``-q`` represent the same rotation. Composition with ``@`` is the
    Hamilton product, so its sign follows from the operands; use ``canonical`` for a
    unique sign.

    Parameters
    ----------
    value : FloatArray
        Quaternion components with shape (4,), ordered by ``format``.
    format : QuaternionFormat
        Order of the components.

    Raises
    ------
    ValueError
        If ``value`` does not have shape (4,) or is not normalized within an absolute
        tolerance of 1e-6.
    """

    value: FloatArray
    format: QuaternionFormat

    def __post_init__(self) -> None:
        """Validate the quaternion and store a read-only float64 copy of it."""
        components = Quaternion._as_components(self.value)
        norm = float(np.linalg.norm(components))
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(
                f"Invalid quaternion: must be normalized, got norm {norm}."
            )
        object.__setattr__(self, "value", to_readonly_array(components))

    @classmethod
    def from_unnormalized_components(
        cls, components: npt.ArrayLike, format: QuaternionFormat
    ) -> Self:
        """
        Create a quaternion by normalizing non-unit components.

        Parameters
        ----------
        components : array_like
            Non-zero quaternion with shape (4,).
        format : QuaternionFormat
            Order of ``components``, kept by the result.

        Raises
        ------
        ValueError
            If ``components`` does not have shape (4,) or has zero length.
        """
        array = Quaternion._as_components(components)
        norm = float(np.linalg.norm(array))
        if np.isclose(norm, 0.0):
            raise ValueError("Invalid quaternion: must have non-zero length.")
        return cls(value=array / norm, format=format)

    @classmethod
    def from_rotation(
        cls,
        rotation: SingleRotation,
        format: QuaternionFormat = QuaternionFormat.WXYZ,
    ) -> Self:
        """
        Convert a rotation from any representation into a quaternion.

        Parameters
        ----------
        rotation : SingleRotation
            Rotation to convert. A quaternion is reordered into ``format``.
        format : QuaternionFormat, default QuaternionFormat.WXYZ
            Order of the components of the result.
        """
        return super().from_rotation(rotation).with_format(format)

    @classmethod
    def identity(cls, format: QuaternionFormat = QuaternionFormat.WXYZ) -> Self:
        """
        Create the identity rotation.

        Parameters
        ----------
        format : QuaternionFormat, default QuaternionFormat.WXYZ
            Order of the components of the result.
        """
        return super().identity().with_format(format)

    @classmethod
    def random(
        cls,
        generator: np.random.Generator,
        format: QuaternionFormat = QuaternionFormat.WXYZ,
    ) -> Self:
        """
        Draw a rotation uniformly at random from SO(3).

        Parameters
        ----------
        generator : numpy.random.Generator
            Source of randomness, such as ``np.random.default_rng(seed)``.
        format : QuaternionFormat, default QuaternionFormat.WXYZ
            Order of the components of the result.
        """
        return super().random(generator).with_format(format)

    @classmethod
    def align_vectors(
        cls,
        sources: npt.ArrayLike,
        targets: npt.ArrayLike,
        weights: npt.ArrayLike | None = None,
        format: QuaternionFormat = QuaternionFormat.WXYZ,
    ) -> Self:
        """
        Estimate the rotation that best maps ``sources`` onto ``targets``.

        See ``SingleRotation.align_vectors`` for the method.

        Parameters
        ----------
        sources : array_like
            Vectors before the rotation, with shape (M, 3).
        targets : array_like
            Corresponding vectors after the rotation, with shape (M, 3).
        weights : array_like or None, default None
            Non-negative weight of each pair with shape (M,). ``None`` weighs every
            pair equally.
        format : QuaternionFormat, default QuaternionFormat.WXYZ
            Order of the components of the result.

        Raises
        ------
        ValueError
            If the shapes or weights are invalid.
        """
        return super().align_vectors(sources, targets, weights).with_format(format)

    def with_format(self, format: QuaternionFormat) -> Self:
        """
        Return the same quaternion with its components reordered into ``format``.

        Parameters
        ----------
        format : QuaternionFormat
            Order of the components of the result.
        """
        return type(self)(value=format.from_wxyz(self.wxyz), format=format)

    @property
    def is_scalar_first(self) -> bool:
        """Return whether ``value`` stores the scalar component first."""
        return self.format.is_scalar_first

    @property
    def wxyz(self) -> FloatArray:
        """Return the components in scalar-first order."""
        return self.format.to_wxyz(self.value)

    @property
    def xyzw(self) -> FloatArray:
        """Return the components in scalar-last order."""
        return QuaternionFormat.XYZW.from_wxyz(self.wxyz)

    @property
    def scalar_part(self) -> float:
        """Return the scalar component ``w``, the cosine of half the angle."""
        return cast(float, self.wxyz[0])

    @property
    def vector_part(self) -> FloatArray:
        """Return ``[x, y, z]``, the axis scaled by the sine of half the angle."""
        return self.wxyz[1:]

    def canonical(self) -> Self:
        """
        Return the quaternion of the same rotation with a non-negative scalar part.

        When the scalar part is zero, the first non-zero component is made positive,
        so every rotation has exactly one canonical quaternion. The format is kept.
        """
        canonical = self._rotation.as_quat(canonical=True, scalar_first=True)
        return type(self)(value=self.format.from_wxyz(canonical), format=self.format)

    def _to_scipy(self) -> Rotation:
        return Rotation.from_quat(np.array(self.wxyz), scalar_first=True)

    @classmethod
    def _from_scipy(cls, rotation: Rotation) -> Self:
        return cls(
            value=rotation.as_quat(scalar_first=True), format=QuaternionFormat.WXYZ
        )

    def _with_rotation(self, rotation: Rotation) -> Self:
        return self._from_scipy(rotation).with_format(self.format)

    @staticmethod
    def _as_components(value: npt.ArrayLike) -> FloatArray:
        components = np.asarray(value, dtype=np.float64)
        if components.shape != (4,):
            raise ValueError(
                f"Invalid quaternion: must be a shape (4,) array, got shape {components.shape}."
            )
        return components
