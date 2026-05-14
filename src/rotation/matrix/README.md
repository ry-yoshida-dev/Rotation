# matrix

## Overview

3×3 rotation matrices in SO(3). `RotationMatrix` is a frozen dataclass wrapping a `numpy.ndarray` of shape `(3, 3)`. Construction runs validation: orthogonal columns, determinant +1.

Composition uses `@` and returns a new instance of the same concrete type. Factory helpers build from the identity or from noisy matrices via QR or SVD projection.

## Components

| Component | Description |
|-----------|-------------|
| [container.py](./container.py) | `RotationMatrix` — `value`, validation in `__post_init__`, `rotation_vector`, transpose / inverse as raw arrays |
| [protocol.py](./protocol.py) | `RotationMatrixLike`, `RotationMatrixType`, `rotation_matrix_ctor` — typing for mixins and `type(self)` factories |
| [mixin/](./mixin/README.md) | Axis accessors, classmethod factories, and `__matmul__` |

See the parent package [../README.md](../README.md) for context.
