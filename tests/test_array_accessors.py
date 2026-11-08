"""Tests for the raw array accessors every representation shares."""

from __future__ import annotations

import numpy as np
import pytest

from rotation import (
    BatchRotation,
    Quaternion,
    QuaternionFormat,
    Quaternions,
    RotationMatrices,
    RotationMatrix,
    RotationVector,
    RotationVectors,
    SingleRotation,
)

HALF = 0.7071067811865476  # cos(π/4) = sin(π/4)


@pytest.mark.parametrize("representation", [RotationMatrix, RotationVector, Quaternion])
def test_single_accessors(representation: type[SingleRotation]) -> None:
    """
    Verify every single representation returns each array form of 90° about +Z.

    Output: (3, 3) matrix, (3,) vector, and (4,) quaternions in both orders.
    """
    source = RotationVector(value=np.array([0.0, 0.0, np.pi / 2]))
    rotation = representation.from_rotation(source)
    np.testing.assert_allclose(
        rotation.rotation_matrix,
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(rotation.rotation_vector, [0.0, 0.0, np.pi / 2])
    np.testing.assert_allclose(rotation.wxyz, [HALF, 0.0, 0.0, HALF])
    np.testing.assert_allclose(rotation.xyzw, [0.0, 0.0, HALF, HALF])


@pytest.mark.parametrize(
    "representation", [RotationMatrices, RotationVectors, Quaternions]
)
def test_batch_accessors_have_leading_batch_axis(
    representation: type[BatchRotation[SingleRotation]],
) -> None:
    """
    Verify batch accessors stack one array per rotation.

    Output: shapes (N, 3, 3), (N, 3), (N, 4), and (N, 4), also for N = 0.
    """
    for count in (0, 4):
        rotations = representation.random(count, np.random.default_rng(0))
        assert rotations.rotation_matrix.shape == (count, 3, 3)
        assert rotations.rotation_vector.shape == (count, 3)
        assert rotations.wxyz.shape == (count, 4)
        assert rotations.xyzw.shape == (count, 4)


def test_own_representation_returns_stored_value() -> None:
    """
    Verify the accessor of a class's own representation is its stored value.

    Input: a rotation vector longer than π and a quaternion with negative scalar.
    Output: the values unchanged, not reduced or canonicalized.
    """
    vector = RotationVector(value=np.array([0.0, 0.0, 1.5 * np.pi]))
    np.testing.assert_array_equal(vector.rotation_vector, vector.value)
    quaternion = Quaternion(
        value=np.array([0.0, 0.0, 0.0, -1.0]), format=QuaternionFormat.XYZW
    )
    np.testing.assert_array_equal(quaternion.xyzw, quaternion.value)
    np.testing.assert_array_equal(quaternion.wxyz, [-1.0, 0.0, 0.0, 0.0])
    matrices = RotationMatrices.identity(2)
    np.testing.assert_array_equal(matrices.rotation_matrix, matrices.value)
