# matrix

## Overview

3×3 rotation matrices in SO(3). `RotationMatrix` is a frozen dataclass wrapping a `numpy.ndarray` of shape `(3, 3)`. Construction runs validation: orthogonal columns, determinant +1.

Composition uses `@` and returns a new instance of the same concrete type. Factory helpers build from the identity or from noisy matrices via QR or SVD projection.

## Components

| Component | Description |
|-----------|-------------|
| [container.py](./container.py) | `RotationMatrix` — `value`, validation in `__post_init__`, `rotation_vector`, transpose / inverse as raw arrays |
| [protocols/](./protocols/README.md) | `RotationMatrixLike`, `RotationMatrixType`, `rotation_matrix_ctor` — typing for mixins and `type(self)` factories |
| [mixin/](./mixin/README.md) | Axis accessors, classmethod factories, and `__matmul__` |

See the parent package [../README.md](../README.md) for context.

## Example

```python
import numpy as np

from rotation.matrix import RotationMatrix

# Identity
R = RotationMatrix.unit_matrix()

# Composition (both sides must be RotationMatrix)
Rz = RotationMatrix(
    value=np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
)
R_composed = R @ Rz

# Project a noisy 3×3 matrix toward SO(3)
noisy = Rz.value + 1e-3 * np.random.default_rng(0).standard_normal((3, 3))
R_fit = RotationMatrix.from_approximate_matrix_with_SVD(noisy)

# Column axes (shape (3,))
x = R_fit.x_axis

# Recover rotation vector (ndarray) from a matrix
w = R_fit.rotation_vector
```
