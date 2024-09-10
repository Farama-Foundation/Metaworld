---
layout: "contents"
title: State Space 
firstpage:
---

# State Space


Likewise the [action space](action_space), the state space among the task requires to maintain the same structure that allows current approaches to employ a single policy/model.
Meta-World contains tasks that either require manipulation of a single object with a potentially variable goal postition (e.g., `reach`, `push`,`pick place`) or two objects with a fixed goal postition (e.g., `hammer`, `soccer`, `shelf place`).
To account for such a variability, large parts of the observation space are kept as placeholders, e.g., for the second object, if only one object is avaiable.

The observation array consists of the end-effector's 3D Cartesian position and the compisition of a single object with its goal coordinates or the positons of the first and second object.
This always results in a 9D state vector.

TODO: Provide table
