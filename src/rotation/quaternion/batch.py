from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ..batch import BatchRotation
from ..types import FloatArray
from ..utils import to_readonly_array
from .format import QuaternionFormat
from .single import Quaternion


@dataclass(frozen=True, eq=False)
class Quaternions(BatchRotation[Quaternion]):
    """
    N rotations stored as unit quaternions ``w + xi + yj + zk``.

    Composition with ``@`` is the Hamilton product of each pair; use ``canonical``
    for a unique sign per rotation.

    Parameters
    ----------
    value : FloatArray
        Quaternion components with shape (N, 4) in scalar-first order
        ``[w, x, y, z]``. Use ``from_components`` for scalar-last input.

    Raises
    ------
    ValueError
        If ``value`` does not have shape (N, 4) or a quaternion is not normalized
        within an absolute tolerance of 1e-6.
    """

    value: FloatArray

    def __post_init__(self) -> None:
        """Validate the quaternions and store a read-only float64 copy of them."""
        components = Quaternions._as_components(self.value)
        norms = np.linalg.norm(components, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError(
                f"Invalid quaternions: must be normalized, got norms {norms}."
            )
        object.__setattr__(self, "value", to_readonly_array(components))

    @classmethod
    def from_components(
        cls, components: npt.ArrayLike, format: QuaternionFormat
    ) -> Self:
        """
        Create quaternions from unit components in the given order.

        Parameters
        ----------
        components : array_like
            Unit quaternions with shape (N, 4).
        format : QuaternionFormat
            Order of ``components``.

        Raises
        ------
        ValueError
            If ``components`` does not have shape (N, 4) or a quaternion is not
            normalized.
        """
        array = Quaternions._as_components(components)
        return cls(value=format.to_wxyz(array))

    @classmethod
    def from_unnormalized_components(
        cls, components: npt.ArrayLike, format: QuaternionFormat
    ) -> Self:
        """
        Create quaternions by normalizing components in the given order.

        Parameters
        ----------
        components : array_like
            Non-zero quaternions with shape (N, 4).
        format : QuaternionFormat
            Order of ``components``.

        Raises
        ------
        ValueError
            If ``components`` does not have shape (N, 4) or a quaternion has zero
            length.
        """
        array = Quaternions._as_components(components)
        norms = np.linalg.norm(array, axis=1)
        if np.any(np.isclose(norms, 0.0)):
            raise ValueError("Invalid quaternions: must have non-zero length.")
        return cls(value=format.to_wxyz(array / norms[:, np.newaxis]))

    def components(self, format: QuaternionFormat) -> FloatArray:
        """
        Return the components with shape (N, 4) in the given order.

        Parameters
        ----------
        format : QuaternionFormat
            Order of the returned components.
        """
        return format.from_wxyz(self.value)

    @property
    def wxyz(self) -> FloatArray:
        """Return the components in scalar-first order with shape (N, 4)."""
        return self.value

    @property
    def xyzw(self) -> FloatArray:
        """Return the components in scalar-last order with shape (N, 4)."""
        return QuaternionFormat.XYZW.from_wxyz(self.value)

    @property
    def scalar_parts(self) -> FloatArray:
        """Return the scalar components ``w`` with shape (N,)."""
        return self.value[:, 0]

    @property
    def vector_parts(self) -> FloatArray:
        """Return ``[x, y, z]`` of each quaternion with shape (N, 3)."""
        return self.value[:, 1:]

    def canonical(self) -> Self:
        """
        Return the quaternions of the same rotations with non-negative scalar parts.

        When a scalar part is zero, the first non-zero component is made positive.
        """
        return type(self)(
            value=self._rotation.as_quat(canonical=True, scalar_first=True).reshape(
                -1, 4
            )
        )

    @classmethod
    def _single_type(cls) -> type[Quaternion]:
        return Quaternion

    def _to_scipy(self) -> Rotation:
        return Rotation.from_quat(
            np.array(self.value).reshape(-1, 4), scalar_first=True
        )

    @classmethod
    def _from_scipy(cls, rotation: Rotation) -> Self:
        return cls(value=rotation.as_quat(scalar_first=True).reshape(-1, 4))

    @staticmethod
    def _as_components(value: npt.ArrayLike) -> FloatArray:
        components = np.asarray(value, dtype=np.float64)
        if components.ndim != 2 or components.shape[1] != 4:
            raise ValueError(
                f"Invalid quaternions: must be a shape (N, 4) array, got shape {components.shape}."
            )
        return components
