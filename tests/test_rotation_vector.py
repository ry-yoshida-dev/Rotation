"""Tests for RotationVector."""

from __future__ import annotations

import numpy as np
import pytest

from rotation import RotationMatrix, RotationVector


class TestRotationVectorValidation:
    def test_rejects_invalid_shape(self) -> None:
        """
        Verify non-(3,) vectors are rejected.

        Input: 2-vector.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="shape"):
            _ = RotationVector(value=np.zeros(2))

    def test_rejects_non_finite(self) -> None:
        """
        Verify vectors with NaN are rejected.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="finite"):
            _ = RotationVector(value=np.array([np.nan, 0.0, 0.0]))


class TestRotationVectorFactories:
    def test_from_axis_angle_normalizes_axis(self) -> None:
        """
        Verify from_axis_angle scales the axis to the angle.

        Input: +Z axis of length 2 and π/2 radians.
        Output: vector of length π/2 along +Z.
        """
        rotation = RotationVector.from_axis_angle(axis=[0.0, 0.0, 2.0], angle=np.pi / 2)
        np.testing.assert_allclose(rotation.value, [0.0, 0.0, np.pi / 2])

    @pytest.mark.parametrize("axis", [[0.0, 0.0, 0.0], [0.0, 1.0]])
    def test_from_axis_angle_rejects_invalid_axis(self, axis: list[float]) -> None:
        """
        Verify from_axis_angle rejects zero-length and wrongly shaped axes.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="axis"):
            _ = RotationVector.from_axis_angle(axis=axis, angle=1.0)

    def test_from_rotation_recovers_half_turn(self) -> None:
        """
        Verify conversion handles the π rotation, where the axis is not in the skew part.

        Input: 180° rotation about +X as a matrix.
        Output: vector of length π along the X axis.
        """
        rotation = RotationVector.from_rotation(
            RotationMatrix(value=np.diag([1.0, -1.0, -1.0]))
        )
        np.testing.assert_allclose(np.abs(rotation.value), [np.pi, 0.0, 0.0])


class TestRotationVectorProperties:
    def test_length_is_not_reduced(self) -> None:
        """
        Verify length keeps the stored magnitude while angle is reduced to [0, π].

        Input: 3π/2 about +Z.
        Output: length 3π/2, angle π/2.
        """
        rotation = RotationVector(value=np.array([0.0, 0.0, 1.5 * np.pi]))
        assert rotation.length == pytest.approx(1.5 * np.pi)
        assert rotation.angle == pytest.approx(0.5 * np.pi)

    def test_axis_is_unit(self) -> None:
        """
        Verify axis returns the normalized direction.

        Input: vector (0, 0, 2).
        Output: (0, 0, 1).
        """
        rotation = RotationVector(value=np.array([0.0, 0.0, 2.0]))
        np.testing.assert_allclose(rotation.axis, [0.0, 0.0, 1.0])

    def test_axis_of_identity_raises(self) -> None:
        """
        Verify the axis of the zero vector is rejected.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="undefined"):
            _ = RotationVector.identity().axis
