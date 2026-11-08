from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Generic, Self, TypeVar, cast, overload

import numpy as np
from scipy.spatial.transform import Rotation

from .representation import RotationRepresentation
from .single import SingleRotation
from .types import BoolArray, FloatArray, IndexArray

_SingleRotationT = TypeVar("_SingleRotationT", bound=SingleRotation, covariant=True)


class BatchRotation(RotationRepresentation, ABC, Generic[_SingleRotationT]):
    """
    N rotations in one representation, processed together.

    Every operation is vectorized over the batch. Composing with ``@`` or comparing
    with a single rotation broadcasts it over the batch; two batches must have the
    same length and are paired element by element. Indexing with an integer returns
    the single rotation, and indexing with a slice or an array returns a batch.
    """

    @classmethod
    @abstractmethod
    def _single_type(cls) -> type[_SingleRotationT]:
        """Return the class of one element of the batch."""

    @classmethod
    def from_rotation(cls, rotations: BatchRotation[SingleRotation]) -> Self:
        """
        Convert a batch from any representation into this one.

        Parameters
        ----------
        rotations : BatchRotation
            Rotations to convert.
        """
        return cls._from_scipy(rotations._rotation)

    @classmethod
    def from_rotations(cls, rotations: Sequence[SingleRotation]) -> Self:
        """
        Gather single rotations in any representations into a batch.

        Parameters
        ----------
        rotations : Sequence[SingleRotation]
            Rotations to gather, possibly empty.
        """
        if len(rotations) == 0:
            return cls.identity(0)
        # The stubs leave the result of concatenate untyped.
        concatenated = cast(
            Rotation,
            Rotation.concatenate([rotation._rotation for rotation in rotations]),
        )
        return cls._from_scipy(concatenated)

    @classmethod
    def identity(cls, count: int) -> Self:
        """
        Create ``count`` identity rotations.

        Parameters
        ----------
        count : int
            Number of rotations.

        Raises
        ------
        ValueError
            If ``count`` is negative.
        """
        if count < 0:
            raise ValueError(f"Invalid count: must be non-negative, got {count}.")
        return cls._from_scipy(Rotation.identity(count))

    @property
    def angles(self) -> FloatArray:
        """Return the rotation angle of each rotation in radians, in [0, π], with shape (N,)."""
        return np.asarray(self._rotation.magnitude(), dtype=np.float64)

    def angles_to(
        self, other: SingleRotation | BatchRotation[SingleRotation]
    ) -> FloatArray:
        """
        Return the angle in radians, in [0, π], from each rotation to ``other``.

        Parameters
        ----------
        other : SingleRotation or BatchRotation
            One rotation, compared with every rotation, or a batch of the same length,
            compared element by element.

        Returns
        -------
        FloatArray
            Geodesic distances with shape (N,).

        Raises
        ------
        ValueError
            If ``other`` is a batch of a different length.
        """
        self._check_pairable(other)
        relative = self._rotation.inv() * other._rotation
        return np.asarray(relative.magnitude(), dtype=np.float64)

    def is_close(
        self,
        other: SingleRotation | BatchRotation[SingleRotation],
        *,
        atol: float = 1e-6,
    ) -> BoolArray:
        """
        Return whether each rotation equals ``other`` within an angle of ``atol`` radians.

        Parameters
        ----------
        other : SingleRotation or BatchRotation
            One rotation, compared with every rotation, or a batch of the same length,
            compared element by element.
        atol : float, default 1e-6
            Largest angle in radians between the rotations that counts as close.

        Returns
        -------
        BoolArray
            Comparison results with shape (N,).

        Raises
        ------
        ValueError
            If ``other`` is a batch of a different length.
        """
        return self.angles_to(other) <= atol

    def __matmul__(self, other: SingleRotation | BatchRotation[SingleRotation]) -> Self:
        """
        Return the compositions that apply ``other`` first and then each rotation.

        Raises
        ------
        ValueError
            If ``other`` is a batch of a different length.
        """
        self._check_pairable(other)
        return self._from_scipy(self._rotation * other._rotation)

    def __rmatmul__(self, other: SingleRotation) -> Self:
        """Return the compositions that apply each rotation first and then ``other``."""
        return self._from_scipy(other._rotation * self._rotation)

    def __len__(self) -> int:
        """Return the number of rotations."""
        return len(self.value)

    @overload
    def __getitem__(self, index: int | np.integer) -> _SingleRotationT: ...

    @overload
    def __getitem__(self, index: slice | IndexArray | BoolArray) -> Self: ...

    def __getitem__(
        self, index: int | np.integer | slice | IndexArray | BoolArray
    ) -> _SingleRotationT | Self:
        """Return one rotation for an integer, and a batch for a slice or an array."""
        if isinstance(index, slice | np.ndarray):
            return type(self)(value=self.value[index])
        # NumPy types integer indexing as Any, since the result shape is not tracked.
        return self._single_type()(value=cast(FloatArray, self.value[index]))

    def __iter__(self) -> Iterator[_SingleRotationT]:
        """Iterate over the rotations one by one."""
        single_type = self._single_type()
        for index in range(len(self)):
            yield single_type(value=cast(FloatArray, self.value[index]))

    def _check_pairable(
        self, other: SingleRotation | BatchRotation[SingleRotation]
    ) -> None:
        if isinstance(other, BatchRotation) and len(other) != len(self):
            raise ValueError(
                f"Invalid batch: must have {len(self)} rotations to pair with, got {len(other)}."
            )
