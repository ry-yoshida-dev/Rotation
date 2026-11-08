from __future__ import annotations

from abc import ABC
from typing import Self

import numpy as np
import numpy.typing as npt
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

    @classmethod
    def random(cls, generator: np.random.Generator) -> Self:
        """
        Draw a rotation uniformly at random from SO(3).

        Parameters
        ----------
        generator : numpy.random.Generator
            Source of randomness, such as ``np.random.default_rng(seed)``.
        """
        return cls._from_scipy(cls._random_scipy(generator, ()))

    @classmethod
    def align_vectors(
        cls,
        sources: npt.ArrayLike,
        targets: npt.ArrayLike,
        weights: npt.ArrayLike | None = None,
    ) -> Self:
        """
        Estimate the rotation that best maps ``sources`` onto ``targets``.

        This solves Wahba's problem with the Kabsch algorithm: it returns the rotation
        ``R`` minimizing ``sum(w_i * |t_i - R @ s_i|^2)``. With one pair of vectors,
        it returns the smallest rotation that maps the source direction onto the
        target direction. Use it to align point sets, after subtracting each set's
        centroid.

        Parameters
        ----------
        sources : array_like
            Vectors before the rotation, with shape (M, 3).
        targets : array_like
            Corresponding vectors after the rotation, with shape (M, 3).
        weights : array_like or None, default None
            Non-negative weight of each pair with shape (M,). ``None`` weighs every
            pair equally.

        Raises
        ------
        ValueError
            If ``sources`` and ``targets`` do not have the same shape (M, 3) with
            M >= 1, or if ``weights`` does not have shape (M,), is negative or
            non-finite, or is zero everywhere.
        """
        source_array = np.asarray(sources, dtype=np.float64)
        target_array = np.asarray(targets, dtype=np.float64)
        if (
            source_array.ndim != 2
            or source_array.shape[1] != 3
            or source_array.shape[0] == 0
        ):
            raise ValueError(
                f"Invalid sources: must be a shape (M, 3) array with M >= 1, got shape {source_array.shape}."
            )
        if target_array.shape != source_array.shape:
            raise ValueError(
                f"Invalid targets: must have the shape of sources {source_array.shape}, got shape {target_array.shape}."
            )
        weight_array = cls._as_weights(weights, len(source_array))
        rotation, _ = Rotation.align_vectors(
            target_array, source_array, weights=weight_array
        )
        return cls._from_scipy(rotation)

    @property
    def angle(self) -> float:
        """
        Return the rotation angle in radians, in [0, π].

        ``RotationVector`` returns its stored length instead, which may exceed π.
        """
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
        return self._with_rotation(self._rotation * other._rotation)
