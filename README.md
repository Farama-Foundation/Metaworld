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

- Observation space: 51
  - joint qpos cos: 9
  - joint qpos sin: 9
  - joint qvel: 9
  - end effector pos: 3
  - end effector quat: 4
  - object1 xyz: 3
  - object1 quat: 4
  - object2 xyz: 3
  - object2 quat: 4
  - goal pos: 3
- Action space: 8
  - joint torque: 7
  - gripper torque: 1

### Jaco

DOF = 7
Joints = 13

- Observation space: 63
  - joint qpos cos: 13
  - joint qpos sin: 13
  - joint qvel: 13
  - end effector pos: 3
  - end effector quat: 4
  - object1 xyz: 3
  - object1 quat: 4
  - object2 xyz: 3
  - object2 quat: 4
  - goal pos: 3
- Action space: 8
  - joint torque: 7
  - gripper torque: 1

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

## Environments

- reach
- button-press-topdown
- button-press
- drawer-close
- drawer-open
- window-close
- window-open
- plate-slide
- sweep

### Sawyer Environments

Worked:

- button-press-topdown
- button-press-topdown-wall
- button-press
- button-press-wall
- dial-turn: quite hard
- door-open: crowded
- door-lock
- drawer-close
- drawer-open

- handle-press
- handle-pull
- reach
- plate-slide
- window-close
- window-open

Failed:

- coffee: XML aqpos order
- assembly: too hard
- basketball: too hard
- bin-picking: too hard
- box-close: too hard
- disassemble: too hard
- door-close: conflict initial position
