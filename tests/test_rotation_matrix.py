"""Tests for RotationMatrix."""

from __future__ import annotations

import numpy as np
import pytest

from rotation import RotationMatrix


class TestRotationMatrixValidation:
    def test_converts_to_readonly_float64(
        self, rotation_z_90_matrix: np.ndarray
    ) -> None:
        """
        Verify single-precision input is stored as a read-only float64 array.

        Input: 90° Z rotation as float32.
        Output: stored float64 array that cannot be written.
        """
        rotation = RotationMatrix(value=rotation_z_90_matrix.astype(np.float32))
        assert rotation.value.dtype == np.float64
        assert not rotation.value.flags.writeable

    def test_rejects_invalid_shape(self) -> None:
        """
        Verify non-(3, 3) arrays are rejected.

        Input: 2×2 array.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="shape"):
            _ = RotationMatrix(value=np.eye(2))

    def test_rejects_non_orthogonal_matrix(self) -> None:
        """
        Verify non-orthogonal arrays are rejected.

        Input: shear matrix.
        Output: ValueError.
        """
        shear = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="orthogonal"):
            _ = RotationMatrix(value=shear)

    def test_rejects_reflection(self) -> None:
        """
        Verify reflection matrices are rejected.

        Input: diagonal matrix with determinant -1.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="determinant"):
            _ = RotationMatrix(value=np.diag([1.0, 1.0, -1.0]))


class TestRotationMatrixFactories:
    def test_from_approximate_matrix_by_svd(
        self, rotation_z_90_matrix: np.ndarray
    ) -> None:
        """
        Verify SVD projection recovers a valid rotation close to the input.

        Input: noisy 90° Z rotation.
        Output: matrix close to the clean rotation.
        """
        noise = 1e-3 * np.random.default_rng(0).standard_normal((3, 3))
        fitted = RotationMatrix.from_approximate_matrix_by_svd(
            rotation_z_90_matrix + noise
        )
        np.testing.assert_allclose(fitted.value, rotation_z_90_matrix, atol=1e-2)

    def test_from_approximate_matrix_by_svd_fixes_reflection(self) -> None:
        """
        Verify SVD projection of a reflection returns a proper rotation.

        Input: reflection through the XY plane.
        Output: matrix with determinant +1.
        """
        fitted = RotationMatrix.from_approximate_matrix_by_svd(
            np.diag([1.0, 1.0, -1.0])
        )
        assert np.linalg.det(fitted.value) == pytest.approx(1.0)

    def test_from_approximate_matrix_by_qr_keeps_first_column(
        self, rotation_z_90_matrix: np.ndarray
    ) -> None:
        """
        Verify QR projection keeps the direction of the first column.

        Input: noisy 90° Z rotation.
        Output: first column parallel to the input's first column.
        """
        noisy = rotation_z_90_matrix + 1e-3 * np.random.default_rng(1).standard_normal(
            (3, 3)
        )
        fitted = RotationMatrix.from_approximate_matrix_by_qr(noisy)
        expected = noisy[:, 0] / np.linalg.norm(noisy[:, 0])
        np.testing.assert_allclose(fitted.x_axis, expected, atol=1e-12)

    @pytest.mark.parametrize(
        "factory",
        [
            RotationMatrix.from_approximate_matrix_by_svd,
            RotationMatrix.from_approximate_matrix_by_qr,
        ],
    )
    def test_approximate_factories_reject_invalid_shape(self, factory: object) -> None:
        """
        Verify the projections reject non-(3, 3) input.

        Output: ValueError.
        """
        assert callable(factory)
        with pytest.raises(ValueError, match="shape"):
            _ = factory(np.eye(2))


class TestRotationMatrixAxes:
    def test_axes_match_columns(self, rotation_z_90_matrix: np.ndarray) -> None:
        """
        Verify axis accessors return the matrix columns.

        Input: 90° Z rotation.
        Output: x/y/z axes equal the respective columns.
        """
        rotation = RotationMatrix(value=rotation_z_90_matrix)
        np.testing.assert_allclose(rotation.x_axis, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(rotation.y_axis, [-1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotation.z_axis, [0.0, 0.0, 1.0])
