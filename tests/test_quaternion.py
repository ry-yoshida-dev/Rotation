"""Tests for Quaternion."""

from __future__ import annotations

import numpy as np
import pytest

from rotation.quaternion import Quaternion, QuaternionFormat


class TestQuaternionValidation:
    def test_rejects_invalid_shape(self) -> None:
        """
        Verify non-(4,) quaternions are rejected.

        Input: 3-vector.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="shape"):
            Quaternion(
                value=np.zeros(3, dtype=np.float64),
                format=QuaternionFormat.WXYZ,
            )

    def test_rejects_unnormalized_quaternion(self) -> None:
        """
        Verify non-unit quaternions are rejected.

        Input: vector with norm 2.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="normalized"):
            Quaternion(
                value=np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64),
                format=QuaternionFormat.WXYZ,
            )


class TestQuaternionFormat:
    def test_wxyz_accessor_for_wxyz_storage(self) -> None:
        """
        Verify wxyz returns stored values for WXYZ layout.

        Input: identity quaternion in WXYZ format.
        Output: unchanged 4-vector.
        """
        value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        quaternion = Quaternion(value=value, format=QuaternionFormat.WXYZ)
        np.testing.assert_allclose(quaternion.wxyz, value)

    def test_xyzw_accessor_for_xyzw_storage(self) -> None:
        """
        Verify xyzw returns stored values for XYZW layout.

        Input: identity quaternion in XYZW format.
        Output: unchanged 4-vector.
        """
        value = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        quaternion = Quaternion(value=value, format=QuaternionFormat.XYZW)
        np.testing.assert_allclose(quaternion.xyzw, value)

    def test_format_conversion_between_layouts(self) -> None:
        """
        Verify layout accessors convert between WXYZ and XYZW.

        Input: same rotation stored in both layouts.
        Output: canonical views match after reordering.
        """
        quaternion_wxyz = Quaternion(
            value=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            format=QuaternionFormat.WXYZ,
        )
        quaternion_xyzw = Quaternion(
            value=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            format=QuaternionFormat.XYZW,
        )
        np.testing.assert_allclose(quaternion_wxyz.xyzw, quaternion_xyzw.xyzw)
        np.testing.assert_allclose(quaternion_xyzw.wxyz, quaternion_wxyz.wxyz)

    def test_is_scalar_first(self) -> None:
        """
        Verify scalar-first flag follows the storage format.

        Output: True for WXYZ, False for XYZW.
        """
        quaternion_wxyz = Quaternion(
            value=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            format=QuaternionFormat.WXYZ,
        )
        quaternion_xyzw = Quaternion(
            value=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            format=QuaternionFormat.XYZW,
        )
        assert quaternion_wxyz.is_scalar_first is True
        assert quaternion_xyzw.is_scalar_first is False


class TestQuaternionConversion:
    def test_rotation_matrix_is_identity_for_unit_quaternion(self) -> None:
        """
        Verify identity quaternion maps to identity rotation matrix.

        Output: 3×3 identity matrix with floating dtype.
        """
        quaternion = Quaternion(
            value=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            format=QuaternionFormat.WXYZ,
        )
        rotation_matrix = quaternion.rotation_matrix
        assert np.issubdtype(rotation_matrix.dtype, np.floating)
        np.testing.assert_allclose(rotation_matrix, np.eye(3), atol=1e-6)
