"""Tests for the stored arrays of each container."""

from __future__ import annotations

import numpy as np
import pytest

from rotation import Quaternion, RotationMatrix, RotationVector, SingleRotation


@pytest.mark.parametrize("representation", [RotationMatrix, RotationVector, Quaternion])
def test_value_is_readonly_float64(representation: type[SingleRotation]) -> None:
    """
    Verify every container stores a read-only float64 array.

    Input: identity of each representation.
    Output: float64 dtype and writeable flag disabled.
    """
    rotation = representation.identity()
    value = rotation.value
    assert value.dtype == np.float64
    assert not value.flags.writeable


def test_value_is_copied() -> None:
    """
    Verify mutating the input array does not change the container.

    Output: stored value keeps the original components.
    """
    source = np.zeros(3)
    rotation = RotationVector(value=source)
    source[0] = 1.0
    np.testing.assert_allclose(rotation.value, np.zeros(3))
