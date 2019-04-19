# Metaworld


Metaworld is a multitask manipulation environment that is currently under development. If you’d like to contribute to creating new environments or running algorithms on these environments, please let us know so we can coordinate efforts.


# Basics
Each environment in /metaworld/multiworld/envs/mujoco/sawyer_xyz is an gym environment. To play with the environment, for example, do 


```
 python /metaworld/multiworld/envs/mujoco/sawyer_xyz/button_press_topdown_6dof.py
```


# Pull Request Guidelines


If you would like to contribute *environments*, please submit a pull request. We require the following guidelines for a new environment:


## 1.
Take a look at the spreadsheet to find an environment that doesn’t exist yet. There are task ideas on the spreadsheet.


## 2. Base environment
Your environment should inherit from SawyerXYZEnv in base.py and Multienv in multienv.py. Your class init file should begin like this:


class SawyerExample6DOFEnv(
    SawyerXYZEnv,
    MultitaskEnv,
    Serializable,
    metaclass=abc.ABCMeta,
):
        SawyerXYZEnv.__init__(self, ...)
        MultitaskEnv.__init__(self, ... )




## 3. XML Files
Your environment XML file should include sawyer_xyz_base.xml and share_config.xml


## 4.[a] Action space options
Your environment should implement the following control modes: 3DOF, 4DOF, and 6DOF (quaternion) control. These modes should be controlled by the rotMode class attribute.[b]


## 5. Train a learned policy, and provide a link to the trained policy to the Metaworld spreadsheet.
The task should be learnable with any RL algorithm of your choice.


## 6. (Optional)
Add your name to the Contributions.txt file. The purpose of this file is to track contributors to this project, so in future papers or reports, we can give proper credit to contributors.


## 7. (Optional) Collect a few demonstrations for your task, and add a link to those demonstrations.
This is helpful practice anyway when determining if your new environment is learnable.


## 8. (Optional) We also welcome help contributing to adding textures for the existing environments.
[a]Also say that they need to define a shaped reward function? Or both a shaped reward function and a success rate metric?
[b]Maybe also ask for the contributor to provide a reasonable camera viewpoint for future vision-based experiments?# multiworld
Multitask Environments for RL

## Basic Usage
This library contains a variety of gym `GoalEnv`s.

As a running example, let's say we have a `CarEnv`.
Like normal gym envs, we can do
```
env = CarEnv()
obs = env.reset()
next_obs, reward, done, info = env.step(action)
```

Unlike `Env`s, the observation space of `GoalEnv`s is a dictionary.
```
print(obs)

# Output:
# {
#     'observation': ...,
#     'desired_goal': ...,
#     'achieved_goal': ...,
# }
```
This can make it rather difficult to use these envs with existing RL code, which
usually expects a flat vector.
Hence, we include a wrapper that converts this dictionary-observation env into a
normal "flat" environment:

```
base_env = CarEnv()
env = FlatGoalEnv(base_env, obs_key='observation')
obs = env.reset()  # returns vector in 'observation' key
action = policy_that_takes_in_vector(obs)
```

The observation space of FlatGoalEnv will be the corresponding env of the vector
(e.g. `gym.space.Box`).
**However, the goal is not part of the observation!**
Not giving the goal to the policy might make the task impossible.

We provide two possible solutions to this:

(1) Use the `get_goal` function
```
base_env = CarEnv()
env = FlatGoalEnv(base_env, obs_key='observation')
obs = env.reset()  # returns just the 'observation'
goal = env.get_goal()
action = policy_that_takes_in_two_vectors(obs, goal)
```
(2) Set
`append_goal_to_obs` to `True`.
```
base_env = CarEnv()
env = FlatGoalEnv(
    base_env,
    append_goal_to_obs=True,  # default value is False
)
obs = env.reset()  # returns 'observation' concatenated to `desired_goal`
action = policy_that_takes_in_vector(obs)
```

## Extending Obs/Goals - Debugging and Multi-Modality
One nice thing about using Dict spaces + FlatGoalEnv is that it makes it really
easy to extend and debug.

For example, this repo includes an `ImageMujocoEnv` wrapper which converts
the observation space of a Mujoco GoalEnv into images.
Rather than completely overwriting `observation`, we simply append the
images to the dictionary:

```
base_env = CarEnv()
env = ImageEnv(base_env)
obs = env.reset()

print(obs)

# Output:
# {
#     'observation': ...,
#     'desired_goal': ...,
#     'achieved_goal': ...,
#     'image_observation': ...,
#     'image_desired_goal': ...,
#     'image_achieved_goal': ...,
#     'state_observation': ...,   # CarEnv sets these values by default
#     'state_desired_goal': ...,
#     'state_achieved_goal': ...,
# }
```

This makes it really easy to debug your environment, by e.g. using state-based
observation but image-based goals:
```
base_env = CarEnv()
wrapped_env = ImageEnv(base_env)
env = FlatGoalEnv(
    base_env,
    obs_key='state_observation',
    goal_key='image_desired_goal',
)
```

It also makes multi-model environments really easy to write!
```
base_env = CarEnv()
wrapped_env = ImageEnv(base_env)
wrapped_env = LidarEnv(wrapped_env)
wrapped_env = LanguageEnv(wrapped_env)
env = FlatGoalEnv(
    base_env,
    obs_key=['image_observation', 'lidar_observation'],
    goal_key=['language_desired_goal', 'image_desired_goal'],
)
obs = env.reset()  # image + lidar observation
goal = env.get_goal()  # language + image goal
```

Note that you don't have to use FlatGoalEnv: you can always just use the
environments manually choose the keys that you care about from the
observation.

## WARNING: `compute_reward` incompatibility
The `compute_reward` interface is slightly different from gym's.
Rather than `compute_reward(desired_goal, achieved_goal, info)` our interface is
 `compute_reward(action, observation)`, where the observation is a dictionary.

## Environments
In order to be able to reproduce results as environments change across time, we have the following set of registered environments:

`SawyerReachXYEnv-v1`: A MuJoCo environment with a 7-DoF Sawyer arm reaching goal positions. The end-effector (EE) is constrained to a 2-dimensional rectangle parallel to a table. The action controls EE position through the use of a mocap. The state is the XY position of the EE and the goal is an XY position of the EE.

`SawyerPushAndReachEnvEasy-v0`, `SawyerPushAndReachEnvMedium-v0`, and `SawyerPushAndReachEnvHard-v0`:  A MuJoCo environment with a 7-DoF Sawyer arm and a small puck on a table that the arm must push to a target position. Control is the same as in `SawyerReachXYEnv-v1`.  The state is the XY position of the EE and the XY position of the puck and the goal is an XY position of the EE and an XY position of the puck. The end effector is constrained to only move in the XY plane. Note, these environments are primarily for debugging purposes.

`SawyerPushAndReachArenaEnv-v0`, `SawyerPushAndReachArenaResetFreeEnv-v0`, `SawyerPushAndReachSmallArenaEnv-v0`, and `SawyerPushAndReachSmallArenaResetFreeEnv-v0`: These environments are the exact same as the pushing environments described above with three key differences: 1) the environment is now contained within an arena 2) the environments can be reset free meaning they do not reset the puck position on calls to `env.reset()` 3) the puck position is not clamped to be within the arena. These are the more realistic versions of the pushing environments and should be used for official results. 

`SawyerDoorHookEnv-v0`, and `SawyerDoorHookResetFreeEnv-v0`: A MuJoCo environment with a 7-DoF Sawyer arm with a hook on the end effector and a door with a handle. Control is the same as in `SawyerReachXYEnv-v1`.  The state is the XY position of the EE and the angle of the door and the goal is an XY position of the EE and an angle of the door. The end effector can move in XYZ. In this environment, reset free means that neither the door nor the hand are reset to their initial position on calls to `env.reset()`. Note for this environment, it is recommended to use pre-sampled goals for vision-based tasks since it is not possible to execute `set_to_goal` for many sampled goal positions. 


## Extra features
### `fixed_goal`
The environments also all taken in `fixed_goal` as a parameter, which disables
resampling the goal each time `reset` is called. This can be useful for
debugging: first make sure the env can solve the single-goal case before trying
the multi-goal case.

### `get_diagnostics`
The function `get_diagonstics(rollouts)` returns an `OrderedDict` of potentially
useful numbers to plot/log.
`rollouts` is a list. Each element of the list should be a dictionary describing
a rollout. A dictionary should have the following keys with the corresponding
values:
```
{
    'observations': np array,
    'actions': np array,
    'next_observations': np array,
    'rewards': np array,
    'terminals': np array,
    'env_infos': list of dictionaries returned by step(),
}
```
