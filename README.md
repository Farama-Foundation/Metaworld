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

### MetaWorld

- Observation space: 39
  - current, previous: 18 x 2
    - gripper xyz: 3 (0, 1, 2)
    - gripper distance apart: 1
    - object1 xyz: 3
    - object1 quat: 4
    - object2 xyz: 3
    - object2 quat: 4
  - goal pos: 3
- Action space: 4
  - delta xyz: 3
  - gripper torque: 1 (7)

### Sawyer

DOF = 7
Joints = 9

- Observation space: 49
  - current, previous: 23 x 2
    - joint qpos: 9
    - object1 xyz: 3
    - object1 quat: 4
    - object2 xyz: 3
    - object2 quat: 4
  - goal pos: 3
- Action space: 8
  - joint torque: 7
  - gripper torque: 1

### Jaco

DOF = 6
Joints = 9

- Observation space: 49
  - current, previous: 23 x 2
    - joint qpos: 9
    - object1 xyz: 3
    - object1 quat: 4
    - object2 xyz: 3
    - object2 quat: 4
  - goal pos: 3
- Action space: 9
  - joint torque: 6
  - gripper torque: 3

### Fetch

DOF = 7
Joints = 9

- Observation space: 49
  - current, previous: 23 x 2
    - joint qpos: 9
    - object1 xyz: 3
    - object1 quat: 4
    - object2 xyz: 3
    - object2 quat: 4
  - goal pos: 3
- Action space: 8
  - joint torque: 7
  - gripper torque: 1
