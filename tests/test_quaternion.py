"""Tests for Quaternion and QuaternionFormat."""

from __future__ import annotations

import numpy as np
import pytest

from rotation import Quaternion, QuaternionFormat, RotationVector


class TestQuaternionValidation:
    def test_rejects_invalid_shape(self) -> None:
        """
        Verify non-(4,) quaternions are rejected.

        Input: 3-vector.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="shape"):
            _ = Quaternion(value=np.zeros(3))

    def test_rejects_unnormalized_quaternion(self) -> None:
        """
        Verify non-unit quaternions are rejected.

        Input: vector with norm 2.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="normalized"):
            _ = Quaternion(value=np.array([2.0, 0.0, 0.0, 0.0]))


class TestQuaternionComponents:
    def test_from_components_reorders_scalar_last(self) -> None:
        """
        Verify scalar-last input is stored scalar first.

        Input: identity as [x, y, z, w].
        Output: value [1, 0, 0, 0].
        """
        quaternion = Quaternion.from_components(
            [0.0, 0.0, 0.0, 1.0], QuaternionFormat.XYZW
        )
        np.testing.assert_allclose(quaternion.value, [1.0, 0.0, 0.0, 0.0])

    @pytest.mark.parametrize("format", list(QuaternionFormat))
    def test_components_roundtrip(self, format: QuaternionFormat) -> None:
        """
        Verify components returns what from_components was given.

        Input: arbitrary unit quaternion in each format.
        Output: identical components.
        """
        components = np.array([0.1, 0.2, 0.3, 0.4])
        components /= np.linalg.norm(components)
        quaternion = Quaternion.from_components(components, format)
        np.testing.assert_allclose(quaternion.components(format), components)

    def test_wxyz_and_xyzw(self) -> None:
        """
        Verify the order accessors.

        Input: [w, x, y, z] = [0.5, 0.5, 0.5, 0.5] with distinct signs.
        Output: rolled components.
        """
        quaternion = Quaternion(value=np.array([0.5, -0.5, 0.5, -0.5]))
        np.testing.assert_allclose(quaternion.wxyz, [0.5, -0.5, 0.5, -0.5])
        np.testing.assert_allclose(quaternion.xyzw, [-0.5, 0.5, -0.5, 0.5])
        assert quaternion.scalar_part == pytest.approx(0.5)
        np.testing.assert_allclose(quaternion.vector_part, [-0.5, 0.5, -0.5])

    def test_from_unnormalized_components(self) -> None:
        """
        Verify non-unit input is normalized.

        Input: [2, 0, 0, 0].
        Output: identity.
        """
        quaternion = Quaternion.from_unnormalized_components(
            [2.0, 0.0, 0.0, 0.0], QuaternionFormat.WXYZ
        )
        np.testing.assert_allclose(quaternion.value, [1.0, 0.0, 0.0, 0.0])

    def test_from_unnormalized_components_rejects_zero(self) -> None:
        """
        Verify the zero quaternion is rejected.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="non-zero"):
            _ = Quaternion.from_unnormalized_components(
                np.zeros(4), QuaternionFormat.WXYZ
            )

    def test_is_scalar_first(self) -> None:
        """
        Verify the scalar-first flag of each format.

        Output: True for WXYZ, False for XYZW.
        """
        assert QuaternionFormat.WXYZ.is_scalar_first is True
        assert QuaternionFormat.XYZW.is_scalar_first is False


class TestQuaternionAlgebra:
    def test_matmul_is_hamilton_product(self) -> None:
        """
        Verify composition equals the Hamilton product, sign included.

        Input: i @ j.
        Output: k.
        """
        i = Quaternion(value=np.array([0.0, 1.0, 0.0, 0.0]))
        j = Quaternion(value=np.array([0.0, 0.0, 1.0, 0.0]))
        np.testing.assert_allclose((i @ j).value, [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_inverse_is_conjugate(self) -> None:
        """
        Verify the inverse of a unit quaternion is its conjugate.

        Output: vector part negated, scalar part kept.
        """
        components = np.array([0.5, 0.5, -0.5, 0.5])
        quaternion = Quaternion(value=components)
        np.testing.assert_allclose(
            quaternion.inverse().value, [0.5, -0.5, 0.5, -0.5], atol=1e-12
        )

    def test_canonical_makes_scalar_non_negative(self) -> None:
        """
        Verify canonical flips a quaternion with negative scalar part.

        Input: -identity.
        Output: identity, and the same rotation as the input.
        """
        negative = Quaternion(value=np.array([-1.0, 0.0, 0.0, 0.0]))
        canonical = negative.canonical()
        np.testing.assert_allclose(canonical.value, [1.0, 0.0, 0.0, 0.0])
        assert canonical.is_close(negative)

    def test_from_rotation_of_rotation_vector(self) -> None:
        """
        Verify conversion from a rotation vector gives the half-angle form.

        Input: π/2 about +Z.
        Output: [cos(π/4), 0, 0, sin(π/4)].
        """
        quaternion = Quaternion.from_rotation(
            RotationVector(value=np.array([0.0, 0.0, np.pi / 2]))
        )
        cosine = 0.7071067811865476  # cos(π/4) = sin(π/4)
        expected = np.array([cosine, 0.0, 0.0, cosine])
        np.testing.assert_allclose(quaternion.value, expected)
