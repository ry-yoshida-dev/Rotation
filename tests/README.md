# tests

## Overview

Pytest suite for the `rotation` package.

## Components

| Component | Description |
|-----------|-------------|
| [conftest.py](./conftest.py) | Shared fixtures for the identity and a 90° rotation about +Z. |
| [test_types.py](./test_types.py) | Tests for the floating-point dtype of each container. |
| [test_rotation_matrix.py](./test_rotation_matrix.py) | Tests for `RotationMatrix`. |
| [test_rotation_vector.py](./test_rotation_vector.py) | Tests for `RotationVector`. |
| [test_quaternion.py](./test_quaternion.py) | Tests for `Quaternion`. |
