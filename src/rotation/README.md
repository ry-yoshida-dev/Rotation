# rotation

## Overview

This package provides immutable, validated containers for 3D rotations. Each container stores a read-only copy of its `value`, and equality is identity-based, so compare values with `np.allclose(a.value, b.value)`.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Shared NumPy array type alias for floating-point values. |
| [utils.py](./utils.py) | Read-only array copies stored by every container. |
| [matrix](./matrix/README.md) | 3×3 rotation matrices in SO(3), with factories and composition. |
| [vector](./vector/README.md) | Rotation vectors (axis–angle), with factories and matrix conversion. |
| [quaternion](./quaternion/README.md) | Unit quaternions and their component layout. |
