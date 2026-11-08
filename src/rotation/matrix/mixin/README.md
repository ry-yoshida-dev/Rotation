# mixin

## Overview

Behavior shared by rotation matrix containers, split by concern.

## Components

| Component | Description |
|-----------|-------------|
| [axis.py](./axis.py) | Column axes of the matrix. |
| [factory.py](./factory.py) | Identity, projection of approximate matrices onto SO(3), and validation of raw arrays. |
| [special.py](./special.py) | Composition with `@`. |
