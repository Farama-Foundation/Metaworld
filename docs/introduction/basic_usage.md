---
layout: "contents"
title: Basic Usage
firstpage:
---

# Basic Usage

## Using the benchmark
Here is a list of benchmark environments for meta-RL (ML*) and multi-task-RL (MT*):
* [__ML1__](https://meta-world.github.io/figures/ml1.gif) is a meta-RL benchmark environment which tests few-shot adaptation to goal variation within single task. You can choose to test variation within any of [50 tasks](https://meta-world.github.io/figures/ml45-1080p.gif) for this benchmark.
* [__ML10__](https://meta-world.github.io/figures/ml10.gif) is a meta-RL benchmark which tests few-shot adaptation to new tasks. It comprises 10 meta-train tasks, and 3 test tasks.
* [__ML45__](https://meta-world.github.io/figures/ml45-1080p.gif) is a meta-RL benchmark which tests few-shot adaptation to new tasks. It comprises 45 meta-train tasks and 5 test tasks.
* [__MT10__](https://meta-world.github.io/figures/mt10.gif), __MT1__, and __MT50__ are multi-task-RL benchmark environments for learning a multi-task policy that perform 10, 1, and 50 training tasks respectively. __MT1__ is similar to __ML1__ because you can choose to test variation within any of [50 tasks](https://meta-world.github.io/figures/ml45-1080p.gif) for this benchmark.  In the original Meta-World experiments, we augment MT10 and MT50 environment observations with a one-hot vector which identifies the task. We don't enforce how users utilize task one-hot vectors, however one solution would be to use a Gym wrapper such as [this one](https://github.com/rlworkgroup/garage/blob/master/src/garage/envs/multi_env_wrapper.py)


### Basics
We provide a `Benchmark` API, that allows constructing environments following the [`gymnasium.Env`](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py#L21) interface.

To use a `Benchmark`, first construct it (this samples the tasks allowed for one run of an algorithm on the benchmark).
Then, construct at least one instance of each environment listed in `benchmark.train_classes` and `benchmark.test_classes`.
For each of those environments, a task must be assigned to it using
`env.set_task(task)` from `benchmark.train_tasks` and `benchmark.test_tasks`,
respectively.
`Tasks` can only be assigned to environments which have a key in
`benchmark.train_classes` or `benchmark.test_classes` matching `task.env_name`.


### Seeding a Benchmark Instance
For the purposes of reproducibility, it may be important to you to seed your benchmark instance.
For example, for the ML1 benchmark environment with the 'pick-place-v2' environment, you can do so in the following way:
```python
import metaworld

SEED = 0  # some seed number here
benchmark = metaworld.ML1('pick-place-v2', seed=SEED)
```

### Running ML1 or MT1
```python
import metaworld
import random

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks

env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, terminate, truncate, info = env.step(a)  # Step the environment with the sampled random action
```
__MT1__ can be run the same way except that it does not contain any `test_tasks`


### Running a benchmark
Create an environment with train tasks (ML10, MT10, ML45, or MT50):
```python
import metaworld
import random

ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

training_envs = []
for name, env_cls in ml10.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.train_tasks
                        if task.env_name == name])
  env.set_task(task)
  training_envs.append(env)

for env in training_envs:
  obs = env.reset()  # Reset environment
  a = env.action_space.sample()  # Sample an action
  obs, reward, terminate, truncate, info = env.step(a)  # Step the environment with the sampled random action
```
Create an environment with test tasks (this only works for ML10 and ML45, since MT10 and MT50 don't have a separate set of test tasks):
```python
import metaworld
import random

ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

testing_envs = []
for name, env_cls in ml10.test_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.test_tasks
                        if task.env_name == name])
  env.set_task(task)
  testing_envs.append(env)

for env in testing_envs:
  obs = env.reset()  # Reset environment
  a = env.action_space.sample()  # Sample an action
  obs, reward, terminate, truncate, info = env.step(a)  # Step the environment with the sampled random action
```

## Accessing Single Goal Environments
You may wish to only access individual environments used in the Meta-World benchmark for your research.
We provide constructors for creating environments where the goal has been hidden (by zeroing out the goal in
the observation) and environments where the goal is observable. They are called GoalHidden and GoalObservable
environments respectively.

You can access them in the following way:
```python
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor

import numpy as np

door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]
door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["door-open-v2-goal-hidden"]

env = door_open_goal_hidden_cls()
env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, terminate, truncate, info = env.step(a)  # Step the environment with the sampled random action
assert (obs[-3:] == np.zeros(3)).all() # goal will be zeroed out because env is HiddenGoal

# You can choose to initialize the random seed of the environment.
# The state of your rng will remain unaffected after the environment is constructed.
env1 = door_open_goal_observable_cls(seed=5)
env2 = door_open_goal_observable_cls(seed=5)

env1.reset()  # Reset environment
env2.reset()
a1 = env1.action_space.sample()  # Sample an action
a2 = env2.action_space.sample()
next_obs1, _, _, _, _ = env1.step(a1)  # Step the environment with the sampled random action

next_obs2, _, _, _ = env2.step(a2)
assert (next_obs1[-3:] == next_obs2[-3:]).all() # 2 envs initialized with the same seed will have the same goal
assert not (next_obs2[-3:] == np.zeros(3)).all()   # The env's are goal observable, meaning the goal is not zero'd out

env3 = door_open_goal_observable_cls(seed=10)  # Construct an environment with a different seed
env1.reset()  # Reset environment
env3.reset()
a1 = env1.action_space.sample()  # Sample an action
a3 = env3.action_space.sample()
next_obs1, _, _, _, _ = env1.step(a1)  # Step the environment with the sampled random action
next_obs3, _, _, _, _ = env3.step(a3)

assert not (next_obs1[-3:] == next_obs3[-3:]).all() # 2 envs initialized with different seeds will have different goals
assert not (next_obs1[-3:] == np.zeros(3)).all()   # The env's are goal observable, meaning the goal is not zero'd out

```
