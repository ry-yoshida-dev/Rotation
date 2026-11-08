"""Tests for random sampling, vector alignment, and mean rotations."""

from __future__ import annotations

import numpy as np
import pytest

from rotation import (
    BatchRotation,
    Quaternion,
    Quaternions,
    RotationMatrices,
    RotationMatrix,
    RotationVector,
    RotationVectors,
    SingleRotation,
)


class TestRandom:
    @pytest.mark.parametrize(
        "representation", [RotationMatrix, RotationVector, Quaternion]
    )
    def test_single_is_reproducible(self, representation: type[SingleRotation]) -> None:
        """
        Verify the same seed draws the same rotation.

        Output: close rotations from two generators with one seed.
        """
        first = representation.random(np.random.default_rng(7))
        second = representation.random(np.random.default_rng(7))
        assert first.is_close(second)

    @pytest.mark.parametrize(
        "representation", [RotationMatrices, RotationVectors, Quaternions]
    )
    def test_batch_has_requested_length(
        self, representation: type[BatchRotation[SingleRotation]]
    ) -> None:
        """
        Verify a batch draw returns count rotations.

        Output: length 5.
        """
        assert len(representation.random(5, np.random.default_rng(0))) == 5

    def test_batch_rejects_negative_count(self) -> None:
        """
        Verify a negative count is rejected.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="count"):
            _ = Quaternions.random(-1, np.random.default_rng(0))

    def test_is_uniform(self) -> None:
        """
        Verify the angles follow the uniform distribution on SO(3).

        The angle of a uniform rotation has density (1 - cos θ) / π, so its mean is
        π / 2 + 2 / π.
        Output: sample mean of the angles close to the expected mean.
        """
        angles = Quaternions.random(20000, np.random.default_rng(0)).angles
        assert np.mean(angles) == pytest.approx(np.pi / 2 + 2 / np.pi, abs=0.02)


class TestAlignVectors:
    def test_recovers_rotation(self) -> None:
        """
        Verify Kabsch recovers the rotation between exact correspondences.

        Input: random vectors and their images under a random rotation.
        Output: the original rotation.
        """
        generator = np.random.default_rng(0)
        rotation = RotationMatrix.random(generator)
        sources = generator.standard_normal((10, 3))
        estimate = RotationMatrix.align_vectors(sources, rotation.apply(sources))
        assert estimate.is_close(rotation)

    def test_weights_select_pairs(self) -> None:
        """
        Verify zero weights ignore their pairs.

        Input: one consistent pair and one contradicting pair with zero weight,
        plus a second consistent pair to fix the rotation.
        Output: the rotation of the consistent pairs.
        """
        rotation = RotationVector.from_axis_angle([0.0, 0.0, 1.0], np.pi / 2)
        sources = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        targets = rotation.apply(sources)
        targets[2] = [0.0, 0.0, -1.0]
        estimate = Quaternion.align_vectors(sources, targets, weights=[1.0, 1.0, 0.0])
        assert estimate.is_close(rotation)

    def test_single_pair_gives_smallest_rotation(self) -> None:
        """
        Verify one pair returns the smallest rotation between the directions.

        Input: +X onto +Y.
        Output: 90° about +Z.
        """
        estimate = RotationVector.align_vectors([[1.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]])
        np.testing.assert_allclose(estimate.value, [0.0, 0.0, np.pi / 2], atol=1e-12)

    @pytest.mark.parametrize(
        ("sources", "targets", "weights", "match"),
        [
            (np.zeros((0, 3)), np.zeros((0, 3)), None, "sources"),
            (np.zeros((2, 2)), np.zeros((2, 2)), None, "sources"),
            (np.eye(3), np.eye(3)[:2], None, "targets"),
            (np.eye(3), np.eye(3), [1.0, 1.0], "weights"),
            (np.eye(3), np.eye(3), [1.0, -1.0, 1.0], "weights"),
            (np.eye(3), np.eye(3), [0.0, 0.0, 0.0], "weights"),
        ],
    )
    def test_rejects_invalid_input(
        self,
        sources: np.ndarray,
        targets: np.ndarray,
        weights: list[float] | None,
        match: str,
    ) -> None:
        """
        Verify invalid shapes and weights are rejected.

        Output: ValueError naming the invalid argument.
        """
        with pytest.raises(ValueError, match=match):
            _ = RotationMatrix.align_vectors(sources, targets, weights=weights)


class TestMean:
    def test_mean_of_symmetric_rotations_is_identity(self) -> None:
        """
        Verify opposite rotations about an axis average to the identity.

        Input: ±30° about +Z.
        Output: identity, in the batch's single representation.
        """
        rotations = RotationVectors.from_axes_angles(
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]), [np.pi / 6, -np.pi / 6]
        )
        mean = rotations.mean()
        assert isinstance(mean, RotationVector)
        assert mean.is_close(RotationVector.identity())

    def test_weights_shift_the_mean(self) -> None:
        """
        Verify a weight of zero leaves out its rotation.

        Output: the mean equals the only weighted rotation.
        """
        rotations = Quaternions.random(3, np.random.default_rng(0))
        assert rotations.mean(weights=[0.0, 1.0, 0.0]).is_close(rotations[1])

    def test_mean_ignores_quaternion_sign(self) -> None:
        """
        Verify q and -q count as the same rotation in the mean.

        Output: the mean of q and -q is q.
        """
        quaternion = Quaternion.random(np.random.default_rng(3))
        batch = Quaternions(value=np.stack([quaternion.value, -quaternion.value]))
        assert batch.mean().is_close(quaternion)

    def test_rejects_empty_batch(self) -> None:
        """
        Verify the mean of no rotations is rejected.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="empty"):
            _ = Quaternions.identity(0).mean()

    def test_rejects_invalid_weights(self) -> None:
        """
        Verify weights of the wrong shape are rejected.

        Output: ValueError.
        """
        with pytest.raises(ValueError, match="weights"):
            _ = Quaternions.identity(2).mean(weights=[1.0])
