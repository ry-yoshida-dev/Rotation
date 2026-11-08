# quaternion

## Overview

Unit quaternions stored in either scalar-first (WXYZ) or scalar-last (XYZW) layout. Construction rejects values that are not normalized.

## Components

| Component | Description |
|-----------|-------------|
| [container.py](./container.py) | `Quaternion`, the quaternion container. |
| [format.py](./format.py) | `QuaternionFormat`, the component layout. |

## Examples

```python
import numpy as np

from rotation.quaternion import Quaternion, QuaternionFormat

quaternion = Quaternion(
    value=np.array([0.0, 0.0, 0.0, 1.0]), format=QuaternionFormat.XYZW
)
print(quaternion.wxyz)  # [1.0, 0.0, 0.0, 0.0]
print(quaternion.rotation_matrix)
```
