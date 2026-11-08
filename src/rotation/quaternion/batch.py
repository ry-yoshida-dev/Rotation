from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ..batch import BatchRotation
from ..single import SingleRotation
from ..types import FloatArray
from ..utils import to_readonly_array, vector_norms
from .format import QuaternionFormat
from .single import Quaternion


@dataclass(frozen=True, eq=False)
class Quaternions(BatchRotation[Quaternion]):
    """
    N rotations stored as unit quaternions ``w + xi + yj + zk``.

    ``value`` keeps the component order given by ``format``, and operations,
    indexing, and the mean keep the format. Composition with ``@`` is the Hamilton
    product of each pair; use ``canonical`` for a unique sign per rotation.

    Parameters
    ----------
    value : FloatArray
        Quaternion components with shape (N, 4), ordered by ``format``.
    format : QuaternionFormat
        Order of the components.

    Raises
    ------
    ValueError
        If ``value`` does not have shape (N, 4) or a quaternion is not normalized
        within an absolute tolerance of 1e-6.
    """

    value: FloatArray
    format: QuaternionFormat

    def __post_init__(self) -> None:
        """Validate the quaternions and store a read-only float64 copy of them."""
        components = Quaternions._as_components(self.value)
        norms = vector_norms(components)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError(
                f"Invalid quaternions: must be normalized, got norms {norms}."
            )
        object.__setattr__(self, "value", to_readonly_array(components))

    @classmethod
    def from_unnormalized_components(
        cls, components: npt.ArrayLike, format: QuaternionFormat
    ) -> Self:
        """
        Create quaternions by normalizing non-unit components.

        Parameters
        ----------
        components : array_like
            Non-zero quaternions with shape (N, 4).
        format : QuaternionFormat
            Order of ``components``, kept by the result.

        Raises
        ------
        ValueError
            If ``components`` does not have shape (N, 4) or a quaternion has zero
            length.
        """
        array = Quaternions._as_components(components)
        norms = vector_norms(array)
        if np.any(np.isclose(norms, 0.0)):
            raise ValueError("Invalid quaternions: must have non-zero length.")
        return cls(value=array / norms[:, np.newaxis], format=format)

    @classmethod
    def from_rotation(
        cls,
        rotations: BatchRotation[SingleRotation],
        format: QuaternionFormat = QuaternionFormat.WXYZ,
    ) -> Self:
        """
        Convert a batch from any representation into quaternions.

        Parameters
        ----------
        rotations : BatchRotation
            Rotations to convert. Quaternions are reordered into ``format``.
        format : QuaternionFormat, default QuaternionFormat.WXYZ
            Order of the components of the result.
        """
        return super().from_rotation(rotations).with_format(format)

    @classmethod
    def from_rotations(
        cls,
        rotations: Sequence[SingleRotation],
        format: QuaternionFormat = QuaternionFormat.WXYZ,
    ) -> Self:
        """
        Gather single rotations in any representations into a batch.

        Parameters
        ----------
        rotations : Sequence[SingleRotation]
            Rotations to gather, possibly empty.
        format : QuaternionFormat, default QuaternionFormat.WXYZ
            Order of the components of the result.
        """
        return super().from_rotations(rotations).with_format(format)

    @classmethod
    def identity(
        cls, count: int, format: QuaternionFormat = QuaternionFormat.WXYZ
    ) -> Self:
        """
        Create ``count`` identity rotations.

        Parameters
        ----------
        count : int
            Number of rotations.
        format : QuaternionFormat, default QuaternionFormat.WXYZ
            Order of the components of the result.

        Raises
        ------
        ValueError
            If ``count`` is negative.
        """
        return super().identity(count).with_format(format)

    @classmethod
    def random(
        cls,
        count: int,
        generator: np.random.Generator,
        format: QuaternionFormat = QuaternionFormat.WXYZ,
    ) -> Self:
        """
        Draw ``count`` rotations independently and uniformly at random from SO(3).

        Parameters
        ----------
        count : int
            Number of rotations.
        generator : numpy.random.Generator
            Source of randomness, such as ``np.random.default_rng(seed)``.
        format : QuaternionFormat, default QuaternionFormat.WXYZ
            Order of the components of the result.

        Raises
        ------
        ValueError
            If ``count`` is negative.
        """
        return super().random(count, generator).with_format(format)

    def with_format(self, format: QuaternionFormat) -> Self:
        """
        Return the same quaternions with their components reordered into ``format``.

        Parameters
        ----------
        format : QuaternionFormat
            Order of the components of the result.
        """
        return replace(self, value=format.from_wxyz(self.wxyz), format=format)

    @property
    def wxyz(self) -> FloatArray:
        """Return the components in scalar-first order with shape (N, 4)."""
        return self.format.to_wxyz(self.value)

    @property
    def xyzw(self) -> FloatArray:
        """Return the components in scalar-last order with shape (N, 4)."""
        return QuaternionFormat.XYZW.from_wxyz(self.wxyz)

    @property
    def scalar_parts(self) -> FloatArray:
        """Return the scalar components ``w`` with shape (N,)."""
        return self.wxyz[:, 0]

    @property
    def vector_parts(self) -> FloatArray:
        """Return ``[x, y, z]`` of each quaternion with shape (N, 3)."""
        return self.wxyz[:, 1:]

    def canonical(self) -> Self:
        """
        Return the quaternions of the same rotations with non-negative scalar parts.

        When a scalar part is zero, the first non-zero component is made positive.
        The format is kept.
        """
        canonical = self._rotation.as_quat(canonical=True, scalar_first=True)
        return self._with_value(self.format.from_wxyz(canonical.reshape(-1, 4)))

    def _value_from_scipy(self, rotation: Rotation) -> FloatArray:
        return self.format.from_wxyz(rotation.as_quat(scalar_first=True).reshape(-1, 4))

    def _with_value(self, value: FloatArray) -> Self:
        return replace(self, value=value)

    def _to_scipy(self) -> Rotation:
        return Rotation.from_quat(np.array(self.wxyz).reshape(-1, 4), scalar_first=True)

    @classmethod
    def _from_scipy(cls, rotation: Rotation) -> Self:
        return cls(
            value=rotation.as_quat(scalar_first=True).reshape(-1, 4),
            format=QuaternionFormat.WXYZ,
        )

    def _single_with_value(self, value: FloatArray) -> Quaternion:
        return Quaternion(value=value, format=self.format)

    @staticmethod
    def _as_components(value: npt.ArrayLike) -> FloatArray:
        components = np.asarray(value, dtype=np.float64)
        if components.ndim != 2 or components.shape[1] != 4:
            raise ValueError(
                f"Invalid quaternions: must be a shape (N, 4) array, got shape {components.shape}."
            )
        return components
