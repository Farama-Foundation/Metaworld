---
layout: "contents"
title: Basic Usage
firstpage:
---

# Basic Usage

## Using the benchmark
There are 6 major benchmarks pre-packaged into Meta-World with support for making your own custom benchmarks. The benchmarks are divided into Multi-Task and Meta reinforcement learning benchmarks.

### Multi-Task Benchmarks
The MT1, MT10, and MT50 benchmarks are the Multi-Task Benchmarks. These benchmarks are used to learn a multi-task policy that can learn 1, 10, or 50 training tasks simultaneously. MT1 benchmarks can be created with any of the 50 tasks available in Meta-World.
In the MT10 and MT50 benchmarks, the observations returned by the benchmark will come with one-hot task IDs appended to the state.

### Meta-Learning Benchmarks
The ML1, ML10, and ML45 benchmarks are 3 meta-reinforcement learning benchmarks available in Meta-World. The ML1 benchmark can be used with any of the 50 tasks available in Meta-World.
The ML1 benchmark tests for few-shot adaptation to goal variations within a single task. The ML10 and ML45 both test few-shot adaptation to new tasks. ML10 comprises 10 train tasks with 5 test tasks, while ML45 comprises of 45 training tasks with 5 test tasks.


### MT1
```python
import gymnasium as gym
import metaworld

SEED = 0  # some seed number here
env = gym.make('Meta-World/MT1', env_name='reach-v3', seed=seed)
obs, info = env.reset()

a = env.action_space.sample() # randomly sample an action
obs, reward, truncate, terminate, info = env.step(a) # apply the randomly sampled action
```

### MT10
MT10 has two different versions that can be returned by ```gym.make```. The first version is the synchronous version of the benchmark where all environments are contained within the same process.
For users with limited compute resources, the synchronous option needs the least resources.
```python
import gymnasium as gym
import metaworld

seed = 42

envs = gym.make('Meta-World/MT10', vector_strategy='sync', seed=seed) # this returns a Synchronous Vector Environment with 10 environments

obs, info = envs.reset() # reset all 10 environments

a = env.action_space.sample() # sample an action for each environment

obs, reward, truncate, terminate, info = envs.step(a) # step all 10 environments
```
Alternatively, for users with more compute we also provide the asynchronous version of the MT10 benchmark where each environment is isolated in it's own process and must use inter-process messaging via pipes to communicate.

```python
envs = gym.make_vec('Meta-World/MT10' vector_strategy='async', seed=seed) # this returns an Asynchronous Vector Environment with 10 environments
```

### MT50
MT50 also contains two different versions, a synchronous and an asynchronous version, of the environments.
```python
import gymnasium as gym
import metaworld

seed = 42

envs = gym.make_vec('Meta-World/MT50', vector_strategy='sync', seed=seed) # this returns a Synchronous Vector Environment with 50 environments

obs, info = envs.reset() # reset all 50 environments

a = env.action_space.sample() # sample an action for each environment

obs, reward, truncate, terminate, info = envs.step(a) # step all 50 environments
```

```python
envs = gym.make_vec('Meta-World/MT50', vector_strategy='async', seed=seed) # this returns an Asynchronous Vector Environment with 50 environments
```


## Meta-Learning Benchmarks
Each Meta-reinforcement learning benchmark has training and testing environments. These environments must be created separately as follows.

### ML1
```python
import gymnasium as gym
import metaworld

seed = 42

train_envs = gym.make('Meta-World/ML1-train', env_name='reach-V3', seed=seed)
test_envs = gym.make('Meta-World/ML1-test', env_name='reach-V3', seed=seed)

# training procedure use train_envs
# testing procedure use test_envs

```


### ML10
Similar to the Multi-Task benchmarks, the ML10 and ML45 environments can be run in synchronous or asynchronous modes.


```python
import gymnasium as gym
import metaworld
train_envs = gym.make_vec('Meta-World/ML10-train', vector_strategy='sync', seed=seed)
test_envs = gym.make_vec('Meta-World/ML10-test', vector_strategy='sync', seed=seed)
```


### ML45
```python
import gymnasium as gym
import metaworld

train_envs = gym.make_vec('Meta-World/ML45-train', vector_strategy='sync', seed=seed)
test_envs = gym.make_vec('Meta-World/ML45-test', vector_strategy='sync', seed=seed)
```


## Custom Benchmarks
Finally, we also provide support for creating custom benchmarks by combining any number of Meta-World environments.

The prefix 'mt' will return environments that are goal observable for Multi-Task reinforcement learning, while the prefix 'ml' will return environments that are partially observable for Meta-reinforcement learning.
Like the included MT and ML benchmarks, these environments can also be run in synchronous or asynchronous mode.
In order to create a custom benchmark, the user must provide a list of environment names with the suffix '-V3'.

```python
import gymnasium as gym
import metaworld

envs = gym.make_vec('Meta-World/custom-mt-envs', vector_strategy='sync', envs_list=['env_name_1-v3', 'env_name_2-v3', 'env_name_3-v3'], seed=seed)
envs = gym.make_vec('Meta-World/custom-ml-envs', vector_strategy='sync', envs_list=['env_name_1-v3', 'env_name_2-v3', 'env_name_3-v3'], seed=seed)
```

## Arguments
The gym.make command supports multiple arguments:

| Argument | Usage | Values |
|----------|-------|--------|
|  seed    | The number to seed the random number generator with | None or int |
| max_episode_steps | The maximum number of steps per episode | None or int |
| use_one_hot | Whether the one hot wrapper should be use to add the task ID to the observation | True or False |
| num_tasks | The number of parametric variations to sample (default:50) | int |
| terminate_on_success | Whether to terminate the episode during training when the success signal is seen | True or False|
| vector_strategy | What kind of vector strategy the environments should be wrapped in | 'sync' or 'async' |
| task_select | How parametric variations should be selected | "random" or "pseudorandom" |
| reward_function_version | Use the original reward functions from Meta-World or the updated ones | "v1" or "v2" |
| reward_normalization_method | Apply a reward normalization wrapper | None or 'gymnasium' or 'exponential' |
| render_mode | The render mode of each environment | None or 'human' or 'rgb_array' or 'depth_array' |
| camera_name | The Mujoco name of the camera that should be used to render | 'corner' or 'topview' or 'behindGripper' or 'gripperPOV' or 'corner2' or 'corner3' or 'corner4' |
| camera_id | The Mujoco ID of the camera that should be used to render | int |
