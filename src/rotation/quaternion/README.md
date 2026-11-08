# quaternion

## Overview

Rotations stored as unit quaternions in the component order of their `QuaternionFormat`, scalar first or scalar last. Operations keep the format of the quaternion they are called on.

## Components

| Component | Description |
|-----------|-------------|
| [single.py](./single.py) | `Quaternion`, one unit quaternion. |
| [batch.py](./batch.py) | `Quaternions`, N unit quaternions. |
| [format.py](./format.py) | `QuaternionFormat`, the component order. |

## Examples

```python
import numpy as np

from rotation.quaternion import Quaternion, QuaternionFormat

quaternion = Quaternion(
    value=np.array([0.0, 0.0, 0.0, -1.0]), format=QuaternionFormat.XYZW
)
print(quaternion.wxyz)  # [-1.0, 0.0, 0.0, 0.0]
print(quaternion.canonical().value)  # [0.0, 0.0, 0.0, 1.0], still scalar last
print(quaternion.with_format(QuaternionFormat.WXYZ).value)  # [-1.0, 0.0, 0.0, 0.0]
```
