# tests

## Overview

Pytest suite for the `rotation` package: type aliases, validation, factories, and conversions between rotation representations.

## Components

| Component | Description |
|-----------|-------------|
| [conftest.py](./conftest.py) | Shared fixtures (identity matrix, 90° Z rotation) |
| [test_types.py](./test_types.py) | `FloatArray` and floating dtype checks on containers |
| [test_rotation_matrix.py](./test_rotation_matrix.py) | `RotationMatrix` validation, factories, composition |
| [test_rotation_vector.py](./test_rotation_vector.py) | `RotationVector` factories and matrix conversion |
| [test_quaternion.py](./test_quaternion.py) | `Quaternion` validation, layout accessors, matrix conversion |
| [test_rodrigues.py](./test_rodrigues.py) | `RodriguesRotationParameter` and `SkewSymmetricMatrix` |

## Examples

```bash
cd /path/to/Rotation
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
