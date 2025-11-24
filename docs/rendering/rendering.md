# Rendering

Each Meta-World environment uses Gymnasium to handle the rendering functions following the [`gymnasium.MujocoEnv`](https://github.com/Farama-Foundation/Gymnasium/blob/94a7909042e846c496bcf54f375a5d0963da2b31/gymnasium/envs/mujoco/mujoco_env.py#L184) interface.

Upon environment creation a user can select a render mode in ```('rgb_array', 'human')```.

For example:

```python
import metaworld
import random

env_name = '' # Pick an environment name

render_mode = '' # set a render mode

env = gym.make('Meta-World/MT1', env_name=env_name, render_mode=render_mode)

obs = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, terminate, truncate, info = env.step(a)  # Step the environment with the sampled random action
```

## Render from a specific camera

In addition to the base render functions, Meta-World supports multiple camera positions.

```python
camera_name = '' # one of: ['corner', 'corner2', 'corner3', 'corner4', 'topview', 'behindGripper', 'gripperPOV']

env = gym.make(env_name=env_name, render_mode=render_mode, camera_name=camera_name)

```

The ID of the camera (from Mujoco) can also be passed.

```python

camera_id = '' # this is an integer that represents the camera ID from Mujoco

env = gym.make(env_name=env_name, render_mode=render_mode, camera_id=camera_id)

```

> Camera views may be rotated in this documentation for optimal presentation.

**corner** or **id: 1**
```{figure} ../_static/rendering/corner.png
   :alt: Camera Name Corner
   :width: 200

```

**corner2** or **id: 2**
```{figure} ../_static/rendering/corner2.png
   :alt: Camera Name Corner2
   :width: 200
```

**corner3** or **id: 3**
```{figure} ../_static/rendering/corner3.png
   :alt: Camera Name Corner3
   :width: 200
```

**topview** or **id: 0**
```{figure} ../_static/rendering/topview.png
   :alt: Camera Name Topview
   :width: 200
```

**behindGripper** or **id: 4**
```{figure} ../_static/rendering/behindGripper.png
   :alt: Camera Name BehindGripper
   :width: 200
```

**gripperPOV** or **id: 5**
```{figure} ../_static/rendering/gripperPOV.png
   :alt: Camera Name GripperPOV
   :width: 200
```
