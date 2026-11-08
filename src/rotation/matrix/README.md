# matrix

## Overview

Rotations stored as 3×3 matrices in SO(3), in the column-vector convention `v_new = R @ v`. Construction rejects matrices that are not orthogonal or whose determinant is not +1.

## Components

| Component | Description |
|-----------|-------------|
| [single.py](./single.py) | `RotationMatrix`, one rotation matrix. |
| [batch.py](./batch.py) | `RotationMatrices`, N rotation matrices. |
| [protocol.py](./protocol.py) | Structural type of rotation matrix containers. |
| [special_orthogonal.py](./special_orthogonal.py) | Validation of matrices and projection of approximate ones onto SO(3) by SVD or QR. |

## Examples

```python
import numpy as np

from rotation.matrix import RotationMatrix

noisy = np.eye(3) + 1e-3 * np.random.default_rng(0).standard_normal((3, 3))
fitted = RotationMatrix.from_approximate_matrix_by_svd(noisy)
print(fitted.z_axis)
```
