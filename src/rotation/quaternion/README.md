# quaternion

## Overview

Rotations stored as unit quaternions in scalar-first order `[w, x, y, z]`. Other component orders are accepted and returned through `QuaternionFormat`.

## Components

| Component | Description |
|-----------|-------------|
| [single.py](./single.py) | `Quaternion`, one unit quaternion. |
| [format.py](./format.py) | `QuaternionFormat`, the component order. |

## Examples

```python
from rotation.quaternion import Quaternion, QuaternionFormat

quaternion = Quaternion.from_components([0.0, 0.0, 0.0, -1.0], QuaternionFormat.XYZW)
print(quaternion.wxyz)  # [-1.0, 0.0, 0.0, 0.0]
print(quaternion.canonical().wxyz)  # [1.0, 0.0, 0.0, 0.0]
```
