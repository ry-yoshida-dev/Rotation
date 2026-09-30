# rotation

## Overview

Utilities for representing 3D rotations: immutable, validated rotation matrices and composition, rotation vectors, and quaternions with configurable layout.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Type aliases for numeric arrays (`FloatArray`) |
| [utils.py](./utils.py) | `to_readonly_array` — read-only copies stored by every container |
| [matrix/](./matrix/README.md) | `RotationMatrix` — frozen dataclass for SO(3) 3×3 matrices, validation, factories, composition (`@`) |
| [vector/](./vector/README.md) | `RotationVector` — axis–angle 3-vector, matrix conversion, factories (`from_matrix`, `from_axis_angle`, …) |
| [quaternion/](./quaternion/README.md) | `Quaternion` / `QuaternionFormat` — normalized quaternions and WXYZ / XYZW ordering |

All containers are frozen dataclasses that store a read-only copy of `value`; mutating the source array after construction does not affect them. Equality and hashing are identity-based (`eq=False`); compare values with `np.allclose(a.value, b.value)`.
