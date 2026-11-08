"""Tests for Quaternion and QuaternionFormat."""

from __future__ import annotations

import numpy as np
import pytest

from rotation import Quaternion, QuaternionFormat, RotationMatrix, RotationVector

WXYZ = QuaternionFormat.WXYZ
XYZW = QuaternionFormat.XYZW


class TestQuaternionValidation:
    def test_rejects_invalid_shape(self) -> None:
        """
        Verify non-(4,) quaternions are rejected.

        Input: 3-vector.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="shape"):
            _ = Quaternion(value=np.zeros(3), format=WXYZ)

    def test_rejects_unnormalized_quaternion(self) -> None:
        """
        Verify non-unit quaternions are rejected.

        Input: vector with norm 2.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="normalized"):
            _ = Quaternion(value=np.array([2.0, 0.0, 0.0, 0.0]), format=WXYZ)


class TestQuaternionFormat:
    def test_value_keeps_given_order(self) -> None:
        """
        Verify value is stored in the given format and the accessors reorder it.

        Input: [x, y, z, w] = [-0.5, 0.5, -0.5, 0.5].
        Output: value unchanged, wxyz rolled.
        """
        components = np.array([-0.5, 0.5, -0.5, 0.5])
        quaternion = Quaternion(value=components, format=XYZW)
        np.testing.assert_array_equal(quaternion.value, components)
        np.testing.assert_allclose(quaternion.xyzw, components)
        np.testing.assert_allclose(quaternion.wxyz, [0.5, -0.5, 0.5, -0.5])
        assert quaternion.scalar_part == pytest.approx(0.5)
        np.testing.assert_allclose(quaternion.vector_part, [-0.5, 0.5, -0.5])

    @pytest.mark.parametrize("format", list(QuaternionFormat))
    def test_same_rotation_in_both_formats(self, format: QuaternionFormat) -> None:
        """
        Verify the format only changes the storage order, not the rotation.

        Input: 90° about +Z in each format.
        Output: +X maps to +Y.
        """
        source = RotationVector(value=np.array([0.0, 0.0, np.pi / 2]))
        quaternion = Quaternion.from_rotation(source, format=format)
        assert quaternion.format is format
        np.testing.assert_allclose(
            quaternion.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12
        )

    def test_with_format(self) -> None:
        """
        Verify with_format reorders the stored components.

        Output: identity stored scalar last.
        """
        quaternion = Quaternion.identity().with_format(XYZW)
        assert quaternion.format is XYZW
        np.testing.assert_allclose(quaternion.value, [0.0, 0.0, 0.0, 1.0])

    def test_operations_keep_format(self) -> None:
        """
        Verify operations on a quaternion return the same format.

        Input: random scalar-last quaternion.
        Output: XYZW for every result.
        """
        quaternion = Quaternion.random(np.random.default_rng(0), format=XYZW)
        results = [
            quaternion.inverse,
            quaternion**0.5,
            quaternion @ RotationMatrix.identity(),
            quaternion.canonical(),
        ]
        assert all(result.format is XYZW for result in results)

    @pytest.mark.parametrize("format", list(QuaternionFormat))
    def test_factories_take_format(self, format: QuaternionFormat) -> None:
        """
        Verify every factory returns the requested format.

        Output: format of each result equals the argument.
        """
        results = [
            Quaternion.identity(format=format),
            Quaternion.random(np.random.default_rng(0), format=format),
            Quaternion.from_rotation(RotationMatrix.identity(), format=format),
            Quaternion.align_vectors(np.eye(3), np.eye(3), format=format),
        ]
        assert all(result.format is format for result in results)

    def test_factories_default_to_scalar_first(self) -> None:
        """
        Verify factories without a format use WXYZ.

        Output: WXYZ.
        """
        assert Quaternion.identity().format is WXYZ

    def test_is_scalar_first(self) -> None:
        """
        Verify the scalar-first flag of each format.

        Output: True for WXYZ, False for XYZW.
        """
        assert WXYZ.is_scalar_first is True
        assert XYZW.is_scalar_first is False
        assert Quaternion.identity(format=XYZW).is_scalar_first is False
        assert WXYZ.as_upper == "WXYZ"
        assert XYZW.as_lower == "xyzw"


class TestQuaternionFactories:
    def test_from_unnormalized_components(self) -> None:
        """
        Verify non-unit input is normalized in the given format.

        Input: [0, 0, 0, 2] scalar last.
        Output: identity stored scalar last.
        """
        quaternion = Quaternion.from_unnormalized_components([0.0, 0.0, 0.0, 2.0], XYZW)
        np.testing.assert_allclose(quaternion.value, [0.0, 0.0, 0.0, 1.0])
        assert quaternion.format is XYZW

    def test_from_unnormalized_components_rejects_zero(self) -> None:
        """
        Verify the zero quaternion is rejected.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="non-zero"):
            _ = Quaternion.from_unnormalized_components(np.zeros(4), WXYZ)

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
        np.testing.assert_allclose(
            quaternion.value, np.array([cosine, 0.0, 0.0, cosine])
        )


class TestQuaternionAlgebra:
    def test_matmul_is_hamilton_product(self) -> None:
        """
        Verify composition equals the Hamilton product, sign included.

        Input: i @ j.
        Output: k.
        """
        i = Quaternion(value=np.array([0.0, 1.0, 0.0, 0.0]), format=WXYZ)
        j = Quaternion(value=np.array([0.0, 0.0, 1.0, 0.0]), format=WXYZ)
        np.testing.assert_allclose((i @ j).value, [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_inverse_is_conjugate(self) -> None:
        """
        Verify the inverse of a unit quaternion is its conjugate.

        Output: vector part negated, scalar part kept.
        """
        quaternion = Quaternion(value=np.array([0.5, 0.5, -0.5, 0.5]), format=WXYZ)
        np.testing.assert_allclose(
            quaternion.inverse.value, [0.5, -0.5, 0.5, -0.5], atol=1e-12
        )

    def test_canonical_makes_scalar_non_negative(self) -> None:
        """
        Verify canonical flips a quaternion with negative scalar part.

        Input: -identity scalar last.
        Output: identity scalar last, the same rotation as the input.
        """
        negative = Quaternion(value=np.array([0.0, 0.0, 0.0, -1.0]), format=XYZW)
        canonical = negative.canonical()
        np.testing.assert_allclose(canonical.value, [0.0, 0.0, 0.0, 1.0])
        assert canonical.is_close(negative)
