"""Tests for rotation numeric array type aliases."""

from __future__ import annotations

import numpy as np

from rotation.matrix import RotationMatrix
from rotation.quaternion import Quaternion, QuaternionFormat
from rotation.rodrigues import RodriguesRotationParameter
from rotation.types import FloatArray
from rotation.vector import RotationVector


class TestFloatArray:
    def test_accepts_floating_ndarray(self) -> None:
        """
        Verify FloatArray accepts floating-point ndarrays.

        Input: float64 vector.
        Output: assignment succeeds and dtype remains floating.
        """
        array: FloatArray = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert np.issubdtype(array.dtype, np.floating)

    def test_rotation_matrix_value_is_floating(self, identity_matrix: np.ndarray) -> None:
        """
        Verify RotationMatrix.value is a floating ndarray.

        Input: identity matrix fixture.
        Output: value dtype is floating.
        """
        rotation_matrix = RotationMatrix(value=identity_matrix)
        assert np.issubdtype(rotation_matrix.value.dtype, np.floating)

    def test_rotation_vector_value_is_floating(self) -> None:
        """
        Verify RotationVector.value is a floating ndarray.

        Input: zero rotation vector.
        Output: value dtype is floating.
        """
        rotation_vector = RotationVector(value=np.zeros(3, dtype=np.float64))
        assert np.issubdtype(rotation_vector.value.dtype, np.floating)

    def test_quaternion_value_is_floating(self) -> None:
        """
        Verify Quaternion.value is a floating ndarray.

        Input: unit quaternion in WXYZ layout.
        Output: value dtype is floating.
        """
        quaternion = Quaternion(
            value=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            format=QuaternionFormat.WXYZ,
        )
        assert np.issubdtype(quaternion.value.dtype, np.floating)

    def test_rodrigues_value_is_floating(self, rotation_z_90_vector: np.ndarray) -> None:
        """
        Verify RodriguesRotationParameter.value is a floating ndarray.

        Input: 90° rotation about +Z.
        Output: value dtype is floating.
        """
        rodrigues = RodriguesRotationParameter(value=rotation_z_90_vector)
        assert np.issubdtype(rodrigues.value.dtype, np.floating)
