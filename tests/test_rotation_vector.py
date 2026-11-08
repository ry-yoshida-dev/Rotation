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
    def test_angle_is_stored_length(self) -> None:
        """
        Verify angle is the stored length, not reduced to [0, π] like other representations.

        Input: 3π/2 about +Z.
        Output: angle 3π/2 for the vector, π/2 for the same rotation as a matrix.
        """
        rotation = RotationVector(value=np.array([0.0, 0.0, 1.5 * np.pi]))
        assert rotation.angle == pytest.approx(1.5 * np.pi)
        assert RotationMatrix.from_rotation(rotation).angle == pytest.approx(
            0.5 * np.pi
        )

    def test_from_matrix_and_zero_vector(
        self, rotation_z_90_matrix: np.ndarray
    ) -> None:
        """
        Verify the array factory and the zero vector.

        Input: 90° Z rotation matrix as an array.
        Output: π/2 about +Z, and a zero vector.
        """
        rotation = RotationVector.from_matrix(rotation_z_90_matrix)
        np.testing.assert_allclose(rotation.value, [0.0, 0.0, np.pi / 2])
        np.testing.assert_array_equal(RotationVector.zero_vector().value, np.zeros(3))

    def test_from_matrix_rejects_reflection(self) -> None:
        """
        Verify from_matrix rejects matrices outside SO(3).

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="determinant"):
            _ = RotationVector.from_matrix(np.diag([1.0, 1.0, -1.0]))

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
