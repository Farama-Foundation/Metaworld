---
layout: "contents"
title: Action Space
firstpage:
---

# Action Space

The action space of the Sawyer robot is a ```Box(-1.0, 1.0, (4,), float32)```.
An action represents the Cartesian displacement dx, dy, and dz of the end effector, and an additional action for gripper control.

| Num | Action | Control Min | Control Max | Name (in XML file) | Joint | Unit |
|-----|--------|-------------|-------------|---------------------|-------|------|
| 0 | Displacement of the end effector in x direction (dx) | -1 | 1 | mocap | N/A | position (m) |
| 1 | Displacement of the end effector in y direction (dy) | -1 | 1 | mocap | N/A | position (m) |
| 2 | Displacement of the end effector in z direction (dz) | -1 | 1 | mocap | N/A | position (m) |
| 3 | Gripper adjustment (closing/opening) | -1 | 1 | rightclaw, leftclaw | r_close, l_close | position (normalized) |
