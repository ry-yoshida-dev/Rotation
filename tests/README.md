# tests

## Overview

Pytest suite for the `rotation` package.

## Components

| Component | Description |
|-----------|-------------|
| [conftest.py](./conftest.py) | Shared fixtures for the identity and a 90° rotation about +Z. |
| [test_types.py](./test_types.py) | Tests for the stored arrays of each container. |
| [test_single_rotation.py](./test_single_rotation.py) | Tests for the operations every single rotation shares. |
| [test_rotation_matrix.py](./test_rotation_matrix.py) | Tests for `RotationMatrix`. |
| [test_rotation_vector.py](./test_rotation_vector.py) | Tests for `RotationVector`. |
| [test_quaternion.py](./test_quaternion.py) | Tests for `Quaternion` and `QuaternionFormat`. |
