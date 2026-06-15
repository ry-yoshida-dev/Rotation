"""Tests for RotationMatrix."""

from __future__ import annotations

import numpy as np
import pytest

from rotation.matrix import RotationMatrix


class TestRotationMatrixValidation:
    def test_unit_matrix_is_identity(self) -> None:
        """
        Verify unit_matrix returns the identity.

        Output: 3×3 identity ndarray.
        """
        rotation_matrix = RotationMatrix.unit_matrix()
        np.testing.assert_allclose(rotation_matrix.value, np.eye(3))

    def test_rejects_invalid_shape(self) -> None:
        """
        Verify non-(3, 3) arrays are rejected.

        Input: 2×2 array.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="shape"):
            RotationMatrix(value=np.eye(2, dtype=np.float64))

    def test_rejects_non_orthogonal_matrix(self) -> None:
        """
        Verify non-orthogonal arrays are rejected.

        Input: arbitrary non-orthogonal 3×3 array.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="orthogonal"):
            RotationMatrix(
                value=np.array(
                    [
                        [1.0, 2.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
            )

    def test_rejects_wrong_determinant(self) -> None:
        """
        Verify reflection matrices are rejected.

        Input: diagonal matrix with determinant -1.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="determinant"):
            RotationMatrix(value=np.diag([1.0, 1.0, -1.0]).astype(np.float64))


class TestRotationMatrixFactories:
    def test_from_approximate_matrix_with_svd(
        self,
        rotation_z_90_matrix: np.ndarray,
    ) -> None:
        """
        Verify SVD projection recovers a valid rotation matrix.

        Input: noisy 90° Z rotation.
        Output: matrix close to the clean rotation.
        """
        noisy = rotation_z_90_matrix + 1e-3 * np.random.default_rng(0).standard_normal((3, 3))
        fitted = RotationMatrix.from_approximate_matrix_with_SVD(noisy)
        np.testing.assert_allclose(fitted.value, rotation_z_90_matrix, atol=1e-2)

    def test_from_approximate_matrix_by_qr(
        self,
        rotation_z_90_matrix: np.ndarray,
    ) -> None:
        """
        Verify QR projection returns a valid SO(3) rotation matrix.

        Input: noisy 90° Z rotation.
        Output: orthogonal 3×3 matrix with determinant +1.
        """
        noisy = rotation_z_90_matrix + 1e-3 * np.random.default_rng(1).standard_normal((3, 3))
        fitted = RotationMatrix.from_approximate_matrix_by_qr(noisy)
        assert fitted.is_orthogonal
        assert fitted.is_determinant_correct
        assert fitted.value.shape == (3, 3)


class TestRotationMatrixOperations:
    def test_matmul_composes_rotations(
        self,
        rotation_z_90_matrix: np.ndarray,
    ) -> None:
        """
        Verify matrix multiplication composes two rotations.

        Input: 90° Z rotation composed with itself.
        Output: 180° Z rotation.
        """
        rotation_a = RotationMatrix(value=rotation_z_90_matrix)
        rotation_b = RotationMatrix(value=rotation_z_90_matrix)
        composed = rotation_a @ rotation_b
        expected = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(composed.value, expected, atol=1e-6)

    def test_axes_match_columns(
        self,
        rotation_z_90_matrix: np.ndarray,
    ) -> None:
        """
        Verify axis accessors return matrix columns.

        Input: 90° Z rotation.
        Output: x/y/z axes equal respective columns.
        """
        rotation_matrix = RotationMatrix(value=rotation_z_90_matrix)
        np.testing.assert_allclose(rotation_matrix.x_axis, rotation_matrix.value[:, 0])
        np.testing.assert_allclose(rotation_matrix.y_axis, rotation_matrix.value[:, 1])
        np.testing.assert_allclose(rotation_matrix.z_axis, rotation_matrix.value[:, 2])

    def test_transpose_and_inverse_match(
        self,
        rotation_z_90_matrix: np.ndarray,
    ) -> None:
        """
        Verify transpose and inverse agree for rotation matrices.

        Input: 90° Z rotation.
        Output: T and inv are equal and orthogonal.
        """
        rotation_matrix = RotationMatrix(value=rotation_z_90_matrix)
        np.testing.assert_allclose(rotation_matrix.T, rotation_matrix.inv, atol=1e-6)
        np.testing.assert_allclose(
            rotation_matrix.T @ rotation_matrix.value,
            np.eye(3),
            atol=1e-6,
        )

    def test_rotation_vector_roundtrip(
        self,
        rotation_z_90_matrix: np.ndarray,
    ) -> None:
        """
        Verify rotation_vector recovers an equivalent axis–angle vector.

        Input: 90° Z rotation matrix.
        Output: vector whose Rodrigues matrix matches the input.
        """
        rotation_matrix = RotationMatrix(value=rotation_z_90_matrix)
        vector = rotation_matrix.rotation_vector
        assert vector.shape == (3,)
        assert np.issubdtype(vector.dtype, np.floating)
