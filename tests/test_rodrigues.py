"""Tests for Rodrigues rotation parameters and skew-symmetric matrices."""

from __future__ import annotations

import numpy as np

from rotation.rodrigues import RodriguesRotationParameter, SkewSymmetricMatrix


class TestSkewSymmetricMatrix:
    def test_from_k_parameter_shape(self) -> None:
        """
        Verify skew matrix has shape (3, 3).

        Input: unit +Z axis components.
        Output: 3×3 floating matrix.
        """
        skew_matrix = SkewSymmetricMatrix.from_k_parameter(k_x=0.0, k_y=0.0, k_z=1.0)
        assert skew_matrix.value.shape == (3, 3)
        assert np.issubdtype(skew_matrix.value.dtype, np.floating)

    def test_squared_is_matrix_product(self) -> None:
        """
        Verify squared returns K @ K.

        Input: skew matrix for +Z.
        Output: product matches explicit multiplication.
        """
        skew_matrix = SkewSymmetricMatrix.from_k_parameter(k_x=0.0, k_y=0.0, k_z=1.0)
        np.testing.assert_allclose(skew_matrix.squared, skew_matrix.value @ skew_matrix.value)


class TestRodriguesRotationParameter:
    def test_transform_rotates_point(
        self,
        rotation_z_90_vector: np.ndarray,
    ) -> None:
        """
        Verify transform applies Rodrigues rotation to a vector.

        Input: 90° rotation about +Z and point on +X.
        Output: rotated point on +Y.
        """
        rodrigues = RodriguesRotationParameter(value=rotation_z_90_vector)
        rotated = rodrigues.transform(np.array([1.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-6)

    def test_rotation_matrix_matches_transform(
        self,
        rotation_z_90_vector: np.ndarray,
    ) -> None:
        """
        Verify rotation_matrix is consistent with transform.

        Input: 90° rotation about +Z.
        Output: matrix-vector product equals transform output.
        """
        rodrigues = RodriguesRotationParameter(value=rotation_z_90_vector)
        point = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        np.testing.assert_allclose(
            rodrigues.rotation_matrix @ point,
            rodrigues.transform(point),
            atol=1e-6,
        )

    def test_zero_rotation_returns_identity(self) -> None:
        """
        Verify near-zero vector yields identity rotation matrix.

        Input: zero Rodrigues vector.
        Output: 3×3 identity matrix.
        """
        rodrigues = RodriguesRotationParameter(
            value=np.zeros(3, dtype=np.float64),
        )
        np.testing.assert_allclose(rodrigues.rotation_matrix, np.eye(3), atol=1e-6)

    def test_component_accessors(
        self,
        rotation_z_90_vector: np.ndarray,
    ) -> None:
        """
        Verify x/y/z accessors return vector components.

        Input: 90° rotation about +Z.
        Output: x=0, y=0, z=π/2.
        """
        rodrigues = RodriguesRotationParameter(value=rotation_z_90_vector)
        assert rodrigues.x == 0.0
        assert rodrigues.y == 0.0
        assert rodrigues.z == np.pi / 2.0
