# rotation

## Overview

Utilities for representing 3D rotations: validated rotation matrices and composition, rotation vectors, quaternions with configurable layout, and Rodrigues parameters with skew-symmetric matrices for conversion.

## Components

| Component | Description |
|-----------|-------------|
| [matrix/](./matrix/README.md) | `RotationMatrix` — frozen dataclass for SO(3) 3×3 matrices, validation, factories, composition (`@`) |
| [vector/](./vector/README.md) | `RotationVector` — axis–angle 3-vector, matrix conversion, factories (`from_matrix`, `from_axis_angle`, …) |
| [quaternion/](./quaternion/README.md) | `Quaternion` / `QuaternionFormat` — normalized quaternions and WXYZ / XYZW ordering |
| [rodrigues/](./rodrigues/README.md) | `RodriguesRotationParameter` / `SkewSymmetricMatrix` — Rodrigues formula and skew-symmetric helpers |

