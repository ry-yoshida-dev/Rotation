# mixin

## Overview

Behavior split across mixins so `RotationMatrix` stays a small container. Each mixin targets `RotationMatrixLike` (`value: np.ndarray`) so subclasses can reuse the same logic with static typing preserved via `RotationMatrixType` / `rotation_matrix_ctor`.

## Components

| Component | Description |
|-----------|-------------|
| [axis.py](./axis.py) | `RotationMatrixAxisMixin` — `x_axis`, `y_axis`, `z_axis` columns as shape `(3,)` |
| [factory.py](./factory.py) | `RotationMatrixFactoryMixin` — `unit_matrix`, `from_approximate_matrix_by_qr`, `from_approximate_matrix_with_SVD`, `validate_rotation_matrix_array` |
| [special.py](./special.py) | `RotationMatrixSpecialMixin` — `__matmul__` composing two rotation matrices |

See the matrix package [../README.md](../README.md) for context.
