"""Shared fixtures for rotation package tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def identity_matrix() -> np.ndarray:
    """
    Return the 3×3 identity rotation matrix.

    Returns
    -------
    np.ndarray
        Identity matrix with dtype float64.
    """
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def rotation_z_90_matrix() -> np.ndarray:
    """
    Return a 90° counter-clockwise rotation about +Z.

    Returns
    -------
    np.ndarray
        Rotation matrix with dtype float64.
    """
    return np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def rotation_z_90_vector() -> np.ndarray:
    """
    Return the axis–angle vector for a 90° rotation about +Z.

    Returns
    -------
    np.ndarray
        Rotation vector with dtype float64.
    """
    return np.array([0.0, 0.0, np.pi / 2.0], dtype=np.float64)
