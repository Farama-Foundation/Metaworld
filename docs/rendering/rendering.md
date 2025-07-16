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

The ID of the camera (from Mujoco) can also be passed if known.

```python

camera_id = '' # this is an integer that represents the camera ID from Mujoco

env = gym.make(env_name=env_name, render_mode=render_mode, camera_id=camera_id)

```
