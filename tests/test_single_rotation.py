"""Tests for the operations every single rotation representation shares."""

from __future__ import annotations

import numpy as np
import pytest

from rotation import Quaternion, RotationMatrix, RotationVector, SingleRotation

REPRESENTATIONS: list[type[SingleRotation]] = [
    RotationMatrix,
    RotationVector,
    Quaternion,
]


@pytest.fixture
def rotation_z_90(rotation_z_90_matrix: np.ndarray) -> RotationMatrix:
    return RotationMatrix(value=rotation_z_90_matrix)


@pytest.fixture
def rotation_x_90() -> RotationVector:
    return RotationVector(value=np.array([np.pi / 2, 0.0, 0.0]))


@pytest.mark.parametrize("representation", REPRESENTATIONS)
class TestSingleRotationAcrossRepresentations:
    def test_identity_leaves_vectors_unchanged(
        self, representation: type[SingleRotation]
    ) -> None:
        """
        Verify identity applies as the identity map.

        Output: unchanged vector and zero angle.
        """
        identity = representation.identity()
        np.testing.assert_allclose(identity.apply([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])
        assert identity.angle == pytest.approx(0.0)

    def test_from_rotation_preserves_rotation(
        self, representation: type[SingleRotation], rotation_z_90: RotationMatrix
    ) -> None:
        """
        Verify conversion into each representation keeps the rotation.

        Input: 90° Z rotation matrix.
        Output: +X maps to +Y.
        """
        converted = representation.from_rotation(rotation_z_90)
        assert isinstance(converted, representation)
        np.testing.assert_allclose(
            converted.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12
        )

    def test_matmul_applies_right_operand_first(
        self,
        representation: type[SingleRotation],
        rotation_z_90: RotationMatrix,
        rotation_x_90: RotationVector,
    ) -> None:
        """
        Verify a @ b rotates by b and then by a, keeping the left representation.

        Input: Z90 @ X90 applied to +Y.
        Output: X90 sends +Y to +Z, and Z90 keeps +Z.
        """
        left = representation.from_rotation(rotation_z_90)
        composed = left @ rotation_x_90
        assert isinstance(composed, representation)
        np.testing.assert_allclose(
            composed.apply([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12
        )

    def test_inverse_undoes_rotation(
        self, representation: type[SingleRotation], rotation_z_90: RotationMatrix
    ) -> None:
        """
        Verify r.inverse() @ r is the identity.

        Output: identity within tolerance.
        """
        rotation = representation.from_rotation(rotation_z_90)
        assert (rotation.inverse() @ rotation).is_close(RotationMatrix.identity())

    def test_power_scales_angle(
        self, representation: type[SingleRotation], rotation_z_90: RotationMatrix
    ) -> None:
        """
        Verify r ** t rotates about the same axis by t times the angle.

        Input: 90° Z rotation to the power 0.5 and 2.
        Output: 45° and 180° about Z.
        """
        rotation = representation.from_rotation(rotation_z_90)
        half = RotationVector.from_axis_angle([0.0, 0.0, 1.0], np.pi / 4)
        double = RotationVector.from_axis_angle([0.0, 0.0, 1.0], np.pi)
        assert (rotation**0.5).is_close(half)
        assert (rotation**2).is_close(double)

    def test_angle_to_is_geodesic_distance(
        self, representation: type[SingleRotation], rotation_z_90: RotationMatrix
    ) -> None:
        """
        Verify angle_to returns the angle of the relative rotation.

        Input: identity and 90° Z rotation.
        Output: π/2.
        """
        identity = representation.identity()
        assert identity.angle_to(rotation_z_90) == pytest.approx(np.pi / 2)

    def test_apply_to_several_vectors(
        self, representation: type[SingleRotation], rotation_z_90: RotationMatrix
    ) -> None:
        """
        Verify apply rotates every row of an (M, 3) array.

        Output: (M, 3) array of rotated rows.
        """
        rotation = representation.from_rotation(rotation_z_90)
        rotated = rotation.apply(np.eye(3))
        np.testing.assert_allclose(rotated, rotation_z_90.value.T, atol=1e-12)

    def test_apply_rejects_invalid_shape(
        self, representation: type[SingleRotation]
    ) -> None:
        """
        Verify apply rejects vectors that are not 3D.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="vectors"):
            _ = representation.identity().apply([1.0, 2.0])


class TestSingleRotationComparison:
    def test_is_close_ignores_representation(
        self, rotation_z_90: RotationMatrix
    ) -> None:
        """
        Verify rotations stored in different representations compare as close.

        Output: True for the same rotation, False for a different one.
        """
        vector = RotationVector.from_rotation(rotation_z_90)
        assert rotation_z_90.is_close(vector)
        assert not rotation_z_90.is_close(RotationMatrix.identity())

    def test_is_close_treats_negated_quaternion_as_equal(self) -> None:
        """
        Verify q and -q are the same rotation.

        Output: True.
        """
        components = np.array([0.5, 0.5, 0.5, 0.5])
        assert Quaternion(value=components).is_close(Quaternion(value=-components))
