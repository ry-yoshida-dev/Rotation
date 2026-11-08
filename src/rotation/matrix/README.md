# matrix

## Overview

3×3 rotation matrices in SO(3). Construction rejects matrices that are not orthogonal or whose determinant is not +1, and composition with `@` returns a new instance of the same class.

## Components

| Component | Description |
|-----------|-------------|
| [container.py](./container.py) | `RotationMatrix`, the rotation matrix container. |
| [protocol.py](./protocol.py) | Structural type that lets the mixins return the concrete subclass. |
| [mixin](./mixin/README.md) | Axis accessors, factories, and composition shared by rotation matrix containers. |

## Examples

```python
import numpy as np

from rotation.matrix import RotationMatrix

noisy = np.eye(3) + 1e-3 * np.random.default_rng(0).standard_normal((3, 3))
fitted = RotationMatrix.from_approximate_matrix_by_svd(noisy)
composed = fitted @ RotationMatrix.unit_matrix()
print(composed.z_axis)
```
