"""Tests for the hat and vee maps between rotation vectors and skew-symmetric matrices."""

from __future__ import annotations

import numpy as np
import pytest

from rotation import (
    RotationMatrix,
    RotationVector,
    RotationVectors,
    SkewSymmetricMatrix,
)


class TestSkewSymmetricMatrix:
    def test_hat_computes_cross_product(self) -> None:
        """
        Verify [ω]× @ v equals ω × v.

        Output: matching cross products.
        """
        omega = np.array([1.0, -2.0, 3.0])
        vector = np.array([0.5, 4.0, -1.0])
        matrix = SkewSymmetricMatrix.from_vectors(omega)
        np.testing.assert_allclose(matrix @ vector, np.cross(omega, vector))
        np.testing.assert_allclose(matrix, -matrix.T)

    def test_vee_inverts_hat_for_stacks(self) -> None:
        """
        Verify vee recovers the vectors of a stack of matrices.

        Output: the original (2, 4, 3) vectors.
        """
        vectors = np.random.default_rng(0).standard_normal((2, 4, 3))
        matrices = SkewSymmetricMatrix.from_vectors(vectors)
        assert matrices.shape == (2, 4, 3, 3)
        np.testing.assert_allclose(SkewSymmetricMatrix.to_vectors(matrices), vectors)

    def test_vee_rejects_non_skew_matrix(self) -> None:
        """
        Verify a symmetric matrix is rejected.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="skew"):
            _ = SkewSymmetricMatrix.to_vectors(np.eye(3))


class TestRotationVectorSkewMatrix:
    def test_exponential_of_skew_matrix_is_rotation_matrix(self) -> None:
        """
        Verify the matrix exponential of [ω]× is the rotation matrix of ω.

        Input: ω = π/2 about +Z, exponentiated by its Taylor series.
        Output: the 90° Z rotation matrix.
        """
        vector = RotationVector.from_axis_angle([0.0, 0.0, 1.0], np.pi / 2)
        skew = vector.skew_matrix
        exponential = np.eye(3)
        term = np.eye(3)
        for order in range(1, 30):
            term = term @ skew / order
            exponential = exponential + term
        np.testing.assert_allclose(
            exponential, RotationMatrix.from_rotation(vector).value, atol=1e-12
        )

    def test_from_skew_matrix(self) -> None:
        """
        Verify from_skew_matrix recovers the vector.

        Output: (1, 2, 3).
        """
        vector = RotationVector(value=np.array([1.0, 2.0, 3.0]))
        recovered = RotationVector.from_skew_matrix(vector.skew_matrix)
        np.testing.assert_allclose(recovered.value, [1.0, 2.0, 3.0])

    def test_from_skew_matrix_rejects_invalid_shape(self) -> None:
        """
        Verify non-(3, 3) input is rejected.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="shape"):
            _ = RotationVector.from_skew_matrix(np.zeros((2, 2)))

    def test_batch_roundtrip(self) -> None:
        """
        Verify the batch hat and vee maps invert each other.

        Output: the original vectors.
        """
        vectors = RotationVectors(
            value=np.random.default_rng(1).standard_normal((5, 3))
        )
        assert vectors.skew_matrices.shape == (5, 3, 3)
        recovered = RotationVectors.from_skew_matrices(vectors.skew_matrices)
        np.testing.assert_allclose(recovered.value, vectors.value)
