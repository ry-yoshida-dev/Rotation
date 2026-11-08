# protocols

## Overview

Structural types for rotation matrix containers and their classes, used by the mixins to keep the concrete subclass in their return types.

## Components

| Component | Description |
|-----------|-------------|
| [like.py](./like.py) | Protocol for objects exposing a read-only rotation matrix `value`. |
| [matrix_type.py](./matrix_type.py) | Protocol for classes constructed with a keyword `value`. |
| [ctor.py](./ctor.py) | Narrows a container class to that constructor protocol. |
