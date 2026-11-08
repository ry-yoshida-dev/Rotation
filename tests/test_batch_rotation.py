"""Tests for the operations every batch rotation representation shares."""

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

REPRESENTATIONS: list[type[BatchRotation[SingleRotation]]] = [
    RotationMatrices,
    RotationVectors,
    Quaternions,
]


@pytest.fixture
def quarter_turns() -> RotationVectors:
    """Return 90° rotations about +X, +Y, and +Z."""
    return RotationVectors(value=np.eye(3) * (np.pi / 2))


@pytest.mark.parametrize("representation", REPRESENTATIONS)
class TestBatchRotationAcrossRepresentations:
    def test_from_rotation_preserves_rotations(
        self,
        representation: type[BatchRotation[SingleRotation]],
        quarter_turns: RotationVectors,
    ) -> None:
        """
        Verify conversion into each representation keeps every rotation.

        Output: each converted rotation is close to the source and +Z maps as expected.
        """
        converted = representation.from_rotation(quarter_turns)
        assert isinstance(converted, representation)
        assert np.all(converted.is_close(quarter_turns))
        np.testing.assert_allclose(
            converted.apply([0.0, 0.0, 1.0]),
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1e-12,
        )

    def test_identity(
        self, representation: type[BatchRotation[SingleRotation]]
    ) -> None:
        """
        Verify identity creates the requested number of zero-angle rotations.

        Output: length 4 and zero angles.
        """
        identities = representation.identity(4)
        assert len(identities) == 4
        np.testing.assert_allclose(identities.angles, np.zeros(4))

    def test_identity_rejects_negative_count(
        self, representation: type[BatchRotation[SingleRotation]]
    ) -> None:
        """
        Verify a negative count is rejected.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="count"):
            _ = representation.identity(-1)

    def test_empty_batch(
        self, representation: type[BatchRotation[SingleRotation]]
    ) -> None:
        """
        Verify an empty batch supports the shared operations.

        Output: empty results with the right trailing shape.
        """
        empty = representation.from_rotations([])
        assert len(empty) == 0
        assert empty.inverse().angles.shape == (0,)
        assert empty.apply([1.0, 0.0, 0.0]).shape == (0, 3)

    def test_indexing(
        self,
        representation: type[BatchRotation[SingleRotation]],
        quarter_turns: RotationVectors,
    ) -> None:
        """
        Verify integers return single rotations and slices or arrays return batches.

        Output: single for 1, batches of the selected length otherwise.
        """
        batch = representation.from_rotation(quarter_turns)
        single = batch[1]
        assert isinstance(single, SingleRotation)
        assert single.is_close(quarter_turns[1])
        assert len(batch[1:]) == 2
        assert len(batch[np.array([0, 2])]) == 2
        assert len(batch[np.array([True, False, True])]) == 2
        assert [rotation.angle for rotation in batch] == pytest.approx([np.pi / 2] * 3)

    def test_matmul_with_single_broadcasts_on_both_sides(
        self,
        representation: type[BatchRotation[SingleRotation]],
        quarter_turns: RotationVectors,
    ) -> None:
        """
        Verify batch @ single and single @ batch compose with every element.

        Output: element-wise compositions in the batch's representation.
        """
        batch = representation.from_rotation(quarter_turns)
        single = RotationMatrix.from_rotation(quarter_turns[2])
        right = batch @ single
        left = single @ batch
        assert isinstance(right, representation)
        assert isinstance(left, representation)
        for index in range(3):
            assert right[index].is_close(quarter_turns[index] @ single)
            assert left[index].is_close(single @ quarter_turns[index])

    def test_matmul_pairs_batches(
        self,
        representation: type[BatchRotation[SingleRotation]],
        quarter_turns: RotationVectors,
    ) -> None:
        """
        Verify batch @ batch composes element by element.

        Input: quarter turns composed with themselves.
        Output: half turns.
        """
        batch = representation.from_rotation(quarter_turns)
        np.testing.assert_allclose((batch @ quarter_turns).angles, [np.pi] * 3)

    def test_rejects_batches_of_different_lengths(
        self,
        representation: type[BatchRotation[SingleRotation]],
        quarter_turns: RotationVectors,
    ) -> None:
        """
        Verify pairing batches of different lengths is rejected.

        Output: ValueError for composition and comparison.
        """
        batch = representation.from_rotation(quarter_turns)
        with pytest.raises(ValueError, match="pair"):
            _ = batch @ quarter_turns[:2]
        with pytest.raises(ValueError, match="pair"):
            _ = batch.angles_to(quarter_turns[:2])

    def test_inverse_and_power(
        self,
        representation: type[BatchRotation[SingleRotation]],
        quarter_turns: RotationVectors,
    ) -> None:
        """
        Verify inverse undoes each rotation and powers scale each angle.

        Output: identities and halved angles.
        """
        batch = representation.from_rotation(quarter_turns)
        assert np.all((batch.inverse() @ batch).is_close(RotationMatrix.identity()))
        np.testing.assert_allclose((batch**0.5).angles, [np.pi / 4] * 3)

    def test_apply_pairs_vectors(
        self,
        representation: type[BatchRotation[SingleRotation]],
        quarter_turns: RotationVectors,
    ) -> None:
        """
        Verify N vectors are rotated by the N rotations pairwise.

        Input: +Y, +Z, +X rotated by quarter turns about X, Y, Z.
        Output: +Z, +X, +Y.
        """
        batch = representation.from_rotation(quarter_turns)
        rotated = batch.apply([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        np.testing.assert_allclose(
            rotated, [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], atol=1e-12
        )


class TestBatchRotationValidation:
    @pytest.mark.parametrize(
        ("representation", "value"),
        [
            (RotationMatrices, np.eye(3)),
            (RotationVectors, np.zeros(3)),
            (Quaternions, np.array([1.0, 0.0, 0.0, 0.0])),
        ],
    )
    def test_rejects_single_shape(
        self, representation: type[BatchRotation[SingleRotation]], value: np.ndarray
    ) -> None:
        """
        Verify a single rotation's array is not accepted as a batch.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="shape"):
            _ = representation(value=value)

    def test_rejects_any_invalid_matrix(self) -> None:
        """
        Verify one reflection in the batch is rejected.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="determinant"):
            _ = RotationMatrices(value=np.stack([np.eye(3), np.diag([1.0, 1.0, -1.0])]))

    def test_rejects_any_unnormalized_quaternion(self) -> None:
        """
        Verify one non-unit quaternion in the batch is rejected.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="normalized"):
            _ = Quaternions(
                value=np.array([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
            )


class TestBatchSpecificAccessors:
    def test_matrix_axes(self, quarter_turns: RotationVectors) -> None:
        """
        Verify the axis accessors return one column per matrix.

        Output: (N, 3) arrays equal to the columns.
        """
        matrices = RotationMatrices.from_rotation(quarter_turns)
        np.testing.assert_allclose(matrices.x_axes, matrices.value[:, :, 0])
        np.testing.assert_allclose(matrices.y_axes, matrices.value[:, :, 1])
        np.testing.assert_allclose(matrices.z_axes, matrices.value[:, :, 2])

    def test_matrix_projections(self) -> None:
        """
        Verify the batch projections return valid rotations close to the input.

        Input: two noisy identities.
        Output: rotations within 1e-2 of the identity.
        """
        noisy = np.eye(3) + 1e-3 * np.random.default_rng(0).standard_normal((2, 3, 3))
        for factory in (
            RotationMatrices.from_approximate_matrices_by_svd,
            RotationMatrices.from_approximate_matrices_by_qr,
        ):
            assert np.all(factory(noisy).angles < 1e-2)

    def test_vectors_from_axes_angles(self) -> None:
        """
        Verify axes are normalized and scaled by their angles.

        Output: rows of the given lengths along the given axes.
        """
        vectors = RotationVectors.from_axes_angles(
            [[0.0, 0.0, 2.0], [3.0, 0.0, 0.0]], [np.pi / 2, np.pi / 4]
        )
        np.testing.assert_allclose(
            vectors.value, [[0.0, 0.0, np.pi / 2], [np.pi / 4, 0.0, 0.0]]
        )
        np.testing.assert_allclose(vectors.axes, [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        np.testing.assert_allclose(vectors.lengths, [np.pi / 2, np.pi / 4])

    def test_vectors_from_axes_angles_rejects_mismatched_angles(self) -> None:
        """
        Verify the number of angles must match the number of axes.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="angles"):
            _ = RotationVectors.from_axes_angles([[0.0, 0.0, 1.0]], [1.0, 2.0])

    def test_quaternion_components_and_canonical(self) -> None:
        """
        Verify scalar-last input is reordered and canonical fixes each sign.

        Output: scalar-first value and non-negative scalar parts.
        """
        quaternions = Quaternions.from_components(
            [[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, 0.0]], QuaternionFormat.XYZW
        )
        np.testing.assert_allclose(quaternions.scalar_parts, [-1.0, 0.0])
        np.testing.assert_allclose(
            quaternions.canonical().wxyz, [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(
            quaternions.xyzw, [[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, 0.0]]
        )

    def test_from_rotations_mixes_representations(self) -> None:
        """
        Verify single rotations in different representations gather into one batch.

        Output: batch whose elements match the inputs.
        """
        singles: list[SingleRotation] = [
            RotationVector.from_axis_angle([0.0, 0.0, 1.0], 1.0),
            Quaternion.identity(),
        ]
        batch = Quaternions.from_rotations(singles)
        assert batch[0].is_close(singles[0])
        assert batch[1].is_close(singles[1])
