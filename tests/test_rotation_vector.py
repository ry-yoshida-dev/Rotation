"""Tests for RotationVector."""

from __future__ import annotations

import numpy as np
import pytest

from rotation.vector import RotationVector


class TestRotationVectorValidation:
    def test_rejects_invalid_shape(self) -> None:
        """
        Verify non-(3,) vectors are rejected.

        Input: 2-vector.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="shape"):
            RotationVector(value=np.zeros(2, dtype=np.float64))


class TestRotationVectorFactories:
    def test_zero_vector_has_zero_angle(self) -> None:
        """
        Verify zero_vector represents identity rotation.

        Output: angle is zero.
        """
        rotation_vector = RotationVector.zero_vector()
        assert rotation_vector.angle == pytest.approx(0.0)

    def test_from_axis_angle(
        self,
        rotation_z_90_vector: np.ndarray,
    ) -> None:
        """
        Verify from_axis_angle builds the expected vector.

        Input: +Z axis and π/2 radians.
        Output: vector aligned with +Z with magnitude π/2.
        """
        rotation_vector = RotationVector.from_axis_angle(
            axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            angle=np.pi / 2.0,
        )
        np.testing.assert_allclose(rotation_vector.value, rotation_z_90_vector, atol=1e-6)

    def test_from_matrix(
        self,
        rotation_z_90_matrix: np.ndarray,
    ) -> None:
        """
        Verify from_matrix extracts an equivalent rotation vector.

        Input: 90° Z rotation matrix.
        Output: vector with angle π/2 about +Z.
        """
        rotation_vector = RotationVector.from_matrix(rotation_z_90_matrix)
        assert rotation_vector.angle == pytest.approx(np.pi / 2.0, abs=1e-6)
        unit_axis = rotation_vector.value / rotation_vector.angle
        np.testing.assert_allclose(unit_axis, np.array([0.0, 0.0, 1.0]), atol=1e-6)


class TestRotationVectorConversion:
    def test_rotation_matrix_rotates_point(
        self,
        rotation_z_90_vector: np.ndarray,
    ) -> None:
        """
        Verify rotation_matrix applies the expected transform.

        Input: 90° rotation about +Z.
        Output: +X point maps to +Y.
        """
        rotation_vector = RotationVector(value=rotation_z_90_vector)
        rotation_matrix = rotation_vector.rotation_matrix
        assert np.issubdtype(rotation_matrix.dtype, np.floating)
        rotated = rotation_matrix @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        np.testing.assert_allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-6)
