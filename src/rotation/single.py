from __future__ import annotations

from abc import ABC
from typing import Self

from scipy.spatial.transform import Rotation

from .representation import RotationRepresentation


class SingleRotation(RotationRepresentation, ABC):
    """
    One rotation, in any representation.

    Representations convert into each other with ``from_rotation``, and compose with
    ``@`` regardless of the representation of the right operand.
    """

    @classmethod
    def from_rotation(cls, rotation: SingleRotation) -> Self:
        """
        Convert a rotation from any representation into this one.

        Parameters
        ----------
        rotation : SingleRotation
            Rotation to convert.
        """
        return cls._from_scipy(rotation._rotation)

    @classmethod
    def identity(cls) -> Self:
        """Create the identity rotation, which leaves every vector unchanged."""
        return cls._from_scipy(Rotation.identity())

    @property
    def angle(self) -> float:
        """Return the rotation angle in radians, in [0, π]."""
        return float(self._rotation.magnitude())

    def angle_to(self, other: SingleRotation) -> float:
        """
        Return the angle in radians, in [0, π], of the rotation from ``self`` to ``other``.

        This is the geodesic distance between the two rotations.

        Parameters
        ----------
        other : SingleRotation
            Rotation to measure the distance to.
        """
        return float((self._rotation.inv() * other._rotation).magnitude())

    def is_close(self, other: SingleRotation, *, atol: float = 1e-6) -> bool:
        """
        Return whether ``other`` is the same rotation within an angle of ``atol`` radians.

        Representations that store the same rotation differently, such as the
        quaternions q and -q, are close.

        Parameters
        ----------
        other : SingleRotation
            Rotation to compare with.
        atol : float, default 1e-6
            Largest angle in radians between the rotations that counts as close.
        """
        return self.angle_to(other) <= atol

    def __matmul__(self, other: SingleRotation) -> Self:
        """
        Return the composition that applies ``other`` first and then ``self``.

        The result has the representation of ``self``.
        """
        if not other._rotation.single:
            # Let a batch on the right broadcast this rotation through its __rmatmul__.
            return NotImplemented
        return self._from_scipy(self._rotation * other._rotation)
