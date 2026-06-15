# mixin

## Overview

Behavior split across mixins so `RotationMatrix` stays a small container. Each mixin targets `RotationMatrixLike` (`value: FloatArray`) so subclasses can reuse the same logic with static typing preserved via `RotationMatrixType` / `rotation_matrix_ctor`.

## Components

| Component | Description |
|-----------|-------------|
| [axis.py](./axis.py) | `RotationMatrixAxisMixin` — `x_axis`, `y_axis`, `z_axis` columns as shape `(3,)` |
| [factory.py](./factory.py) | `RotationMatrixFactoryMixin` — `unit_matrix`, `from_approximate_matrix_by_qr`, `from_approximate_matrix_with_SVD`, `validate_rotation_matrix_array` |
| [special.py](./special.py) | `RotationMatrixSpecialMixin` — `__matmul__` composing two rotation matrices |

See the matrix package [../README.md](../README.md) for context.

## Example

Mixins are composed on `RotationMatrix`; you normally import the concrete class only. The snippet below touches each mixin group: factories, axis accessors, and `@` composition.

```python
import numpy as np

from rotation.matrix import RotationMatrix

R = RotationMatrix.unit_matrix()
assert np.allclose(R.x_axis, np.array([1.0, 0.0, 0.0]))

R_qr = RotationMatrix.from_approximate_matrix_by_qr(
    np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
)

R2 = RotationMatrix(
    value=np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
)
R_prod = R @ R2
```
