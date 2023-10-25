# Metaworld

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Farama-Foundation/metaworld/blob/master/LICENSE)

Forked from Farama-Foundation/metaworld.

## Installation

```bash
export PYTHONPATH=/path/to/metaworld
```

## Usage

```python
from metaworld.envs import ALL_ENVIRONMENTS as ALL_ENV

print(ALL_ENV)
```

## Dimensions

### Sawyer

- Observation space: 61
  - current, previous: 29 x 2
    - gripper xyz: 3 (0, 1, 2) (29, 30, 31)
    - gripper quat: 4 (3, 4, 5, 6) (32, 33, 34, 35)
    - joint qpos: 7 (7, 8, 9, 10, 11, 12, 13) (36, 37, 38, 39, 40, 41, 42)
    - gripper distance apart: 1 (14) (43)
    - object1 xyz: 3 (15, 16, 17) (44, 45, 46)
    - object1 quat: 4 (18, 19, 20, 21) (47, 48, 49, 50)
    - object2 xyz: 3 (22, 23, 24) (51, 52, 53)
    - object2 quat: 4 (25, 26, 27, 28) (54, 55, 56, 57)
  - goal pos: 3 (58, 59, 60)
- Action space: 8
  - joint torque: 7 (0, 1, 2, 3, 4, 5, 6)
  - gripper torque: 1 (7)

### Jaco

- Observation space: 65
  - current, previous: 31 x 2
    - gripper xyz: 3 (0, 1, 2) (31, 32, 33)
    - gripper quat: 4 (3, 4, 5, 6) (34, 35, 36, 37)
    - joint qpos: 9 (7, 8, 9, 10, 11, 12, 13, 14, 15) (38, 39, 40, 41, 42, 43, 44, 45, 46)
    - gripper distance apart: 1 (16) (47)
    - object1 xyz: 3 (17, 18, 19) (48, 49, 50)
    - object1 quat: 4 (20, 21, 22, 23) (51, 52, 53, 54)
    - object2 xyz: 3 (24, 25, 26) (55, 56, 57)
    - object2 quat: 4 (27, 28, 29, 30) (58, 59, 60, 61)
  - goal pos: 3
- Action space: 9
  - joint torque: 6 (0, 1, 2, 3, 4, 5)
  - gripper torque: 3 (6, 7, 8)

### Fetch

- Observation space: 65
  - current, previous: 31 x 2
    - gripper xyz: 3 (0, 1, 2) (31, 32, 33)
    - gripper quat: 4 (3, 4, 5, 6) (34, 35, 36, 37)
    - joint qpos: 9 (7, 8, 9, 10, 11, 12, 13, 14, 15) (38, 39, 40, 41, 42, 43, 44, 45, 46)
    - gripper distance apart: 1 (16) (47)
    - object1 xyz: 3 (17, 18, 19) (48, 49, 50)
    - object1 quat: 4 (20, 21, 22, 23) (51, 52, 53, 54)
    - object2 xyz: 3 (24, 25, 26) (55, 56, 57)
    - object2 quat: 4 (27, 28, 29, 30) (58, 59, 60, 61)
  - goal pos: 3
- Action space: 8
  - joint torque: 7 (0, 1, 2, 3, 4, 5, 6)
  - gripper torque: 1 (7)
