---
layout: "contents"
title: State Space 
firstpage:
---

# State Space

The observation array consists of the gripper's (end effector's) position and state, alongside the object of interest's position and orientation. This table will detail each component usually present in such environments:

| Num | Observation Description                       | Min     | Max     | Site Name (XML)        | Joint Name (XML) | Joint Type | Unit        |
|-----|-----------------------------------------------|---------|---------|------------------------|-------------------|------------|-------------|
| 0   | End effector x position in global coordinates | -Inf    | Inf     | hand                   | -                 | -          | position (m)|
| 1   | End effector y position in global coordinates | -Inf    | Inf     | hand                   | -                 | -          | position (m)|
| 2   | End effector z position in global coordinates | -Inf    | Inf     | hand                   | -                 | -          | position (m)|
| 3   | Gripper distance apart                       | 0.0     | 1.0     | -                      | -                 | -          | dimensionless|
| 4   | Object x position in global coordinates       | -Inf    | Inf     | objGeom (derived)      | -                 | -          | position (m)|
| 5   | Object y position in global coordinates       | -Inf    | Inf     | objGeom (derived)      | -                 | -          | position (m)|
| 6   | Object z position in global coordinates       | -Inf    | Inf     | objGeom (derived)      | -                 | -          | position (m)|
| 7   | Object x quaternion component in global coordinates | -Inf    | Inf | objGeom (derived)      | -                 | -          | quaternion  |
| 8   | Object y quaternion component in global coordinates | -Inf    | Inf | objGeom (derived)      | -                 | -          | quaternion  |
| 9   | Object z quaternion component in global coordinates | -Inf    | Inf | objGeom (derived)      | -                 | -          | quaternion  |
| 10  | Object w quaternion component in global coordinates | -Inf    | Inf | objGeom (derived)      | -                 | -          | quaternion  |
| 11  | Previous end effector x position              | -Inf    | Inf     | hand                   | -                 | -          | position (m)|
| 12  | Previous end effector y position              | -Inf    | Inf     | hand                   | -                 | -          | position (m)| 
| 13  | Previous end effector z position              | -Inf    | Inf     | hand                   | -                 | -          | position (m)|
| 14  | Previous gripper distance apart               | 0.0     | 1.0     | -                      | -                 | -          | dimensionless|
| 15  | Previous object x position in global coordinates | -Inf | Inf     | objGeom (derived)      | -                 | -          | position (m)|
| 16  | Previous object y position in global coordinates | -Inf | Inf     | objGeom (derived)      | -                 | -          | position (m)|
| 17  | Previous object z position in global coordinates | -Inf | Inf     | objGeom (derived)      | -                 | -          | position (m)|
| 18  | Previous object x quaternion component in global coordinates | -Inf | Inf | objGeom (derived) | - | - | quaternion |
| 19  | Previous object y quaternion component in global coordinates | -Inf | Inf | objGeom (derived) | - | - | quaternion |
| 20  | Previous object z quaternion component in global coordinates | -Inf | Inf | objGeom (derived) | - | - | quaternion |
| 21  | Previous object w quaternion component in global coordinates | -Inf | Inf | objGeom (derived) | - | - | quaternion |
| 22  | Goal x position                                | -Inf    | Inf     | goal (derived)         | -                 | -          | position (m)|
| 23  | Goal y position                                | -Inf    | Inf     | goal (derived)         | -                 | -          | position (m)|
| 24  | Goal z position                                | -Inf    | Inf     | goal (derived)         | -                 | -          | position (m)|
