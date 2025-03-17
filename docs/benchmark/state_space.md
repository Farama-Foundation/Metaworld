---
layout: "contents"
title: State Space
firstpage:
---

# State Space

Like the [action space](action_space), the state space among the tasks is maintains the same structure such that a single policy/model can be shared between tasks.
Meta-World contains tasks that either manipulate a single object with a potentially variable goal position (e.g., reach, push, pick place) or to manipulate two objects with a fixed goal position (e.g., hammer, soccer, shelf place).
To account for such variability, large parts of the observation space are kept as placeholders, e.g., for the second object, if only one object is available.

The observation array consists of the end-effector's 3D Cartesian position and the composition of a single object with its goal coordinates or the positions of the first and second object.
This always results in a 9D state vector.

| Indices | Description |
|---------|-------------|
| 0:2 | the XYZ coordinates of the end-effector |
| 3 | a scalar value that represents how open/closed the gripper is |
| 4:6 | the XYZ coordinates of the first object |
| 7:10 | the quaternion describing the spatial orientations and rotations of object #1 |
| 11:13 | the XYZ coordinates of the second object |
| 14:17 | the quaternion describing the spatial orientations and rotations of object #2 |
