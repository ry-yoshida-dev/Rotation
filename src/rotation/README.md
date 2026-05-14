# rotation

## Overview

Utilities for representing 3D rotations: validated rotation matrices and composition, quaternions with configurable layout, and Rodrigues parameters with skew-symmetric matrices for conversion.

## Components

| Component | Description |
|-----------|-------------|
| [matrix/](./matrix/README.md) | `RotationMatrix` — frozen dataclass for SO(3) 3×3 matrices, validation, factories, composition (`@`) |
| [quaternion/](./quaternion/README.md) | `Quaternion` / `QuaternionFormat` — normalized quaternions and WXYZ / XYZW ordering |
| [rodrigues/](./rodrigues/README.md) | `RodriguesRotationParameter` / `SkewSymmetricMatrix` — Rodrigues formula and skew-symmetric helpers |
