# Meta-World
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Farama-Foundation/metaworld/blob/master/LICENSE)
![Build Status](https://github.com/Farama-Foundation/Metaworld/workflows/MetaWorld%20CI/badge.svg)

__Meta-World is an open-source simulated benchmark for meta-reinforcement learning and multi-task learning consisting of 50 distinct robotic manipulation tasks.__ We aim to provide task distributions that are sufficiently broad to evaluate meta-RL algorithms' generalization ability to new behaviors.

For more background information, please refer to our [website](https://metaworld.farama.org/).

__Table of Contents__
- [Installation](#installation)
- [Using the benchmark](#using-the-benchmark)
  * [Basics](#basics)
  * [Seeding a Benchmark Instance](#seeding-a-benchmark-instance)
  * [Running ML1, MT1](#running-ml1-or-mt1)
  * [Running ML10, ML45, MT10, MT50](#running-a-benchmark)
  * [Accessing Single Goal Environments](#accessing-single-goal-environments)
- [Citing Meta-World](#citing-meta-world)
- [Accompanying Baselines](accompanying-baselines)
- [Become a Contributor](#become-a-contributor)
- [Acknowledgements](#acknowledgements)

## Join the Community

Metaworld is now maintained by the Farama Foundation! You can interact with our community and the new developers in our [Discord server](https://discord.gg/PfR7a79FpQ)

## Maintenance Status
The current roadmap for Meta-World can be found [here](https://github.com/Farama-Foundation/Metaworld/issues/409)

## Installation
To install everything, run:

```
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
```

Alternatively, you can clone the repository and install an editable version locally:

```sh
git clone https://github.com/Farama-Foundation/Metaworld.git
cd Metaworld
pip install -e .
```

## Using the benchmark
Here is a list of benchmark environments for meta-RL (ML*) and multi-task-RL (MT*):
* ML1 is a meta-RL benchmark which tests few-shot adaptation to goal variations within a single task. It comprises 1 train task and 1 test tasks. 
* ML10 is a meta-RL benchmark which tests few-shot adaptation to new tasks. It comprises 10 meta-train tasks, and 5 test tasks.
* ML45 is a meta-RL benchmark which tests few-shot adaptation to new tasks. It comprises 45 meta-train tasks and 5 test tasks.
* MT1 is a benchmark for learning a policy for single tasks with multiple goals. It comprises 1 train task and 0 test tasks.
* MT10 is a benchmark for learning a policy for multiple tasks with multiple goals. It comprises 10 train task and 0 test tasks.
* MT50 is a benchmark for learning a policy for multiple tasks with multiple goals. It comprises 50 train task and 0 test tasks.

To view all available environment variations:

```python
import gymnasium as gym
import metaworld

gym.envs.pprint_registry()
```


### Basics
We provide environments via gym.make

You may wish to only access individual environments used in the Metaworld benchmark for your research. See the
[Accessing Single Goal Environments](#accessing-single-goal-environments) for more details.


### Seeding a Benchmark Instance
For the purposes of reproducibility, it may be important to you to seed your benchmark instance.
For example, for the ML1 benchmark with the 'pick-place-v2' environment, you can do so in the following way:
```python
import metaworld

SEED = 0  # some seed number here
env = gym.make('ML-pick-place-v2-train', seed=SEED)
```

### Running ML1 or MT1
```python
import gymnasium as gym
import metaworld
import random

gym.envs.pprint_registry() # print all available environments (this includes environments in Gymnasium)

env = gym.make('ML-pick-place-train', seed=SEED)

obs, info = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, terminate, truncate, info = env.step(a)  # Step the environment with the sampled random action

```
__MT1__ can be run the same way except that it does not contain any `test_tasks`
### Running a benchmark
Create an environment with train tasks (ML10, MT10, ML45, or MT50):
```python
import gymnasium as gym
import metaworld
import random

train_envs = gym.make('ML10-train', seed=SEED)

obs, info = train_envs.reset()  # Reset environment
a = train_envs.action_space.sample()  # Sample an action

obs, reward, terminate, truncate, info = train_envs.step(a)  # Step all environments with the sampled random actions
```
Create an environment with test tasks (this only works for ML10 and ML45, since MT10 and MT50 don't have a separate set of test tasks):
```python
import gymnasium as gym
import metaworld
import random

test_envs = gym.make('ML10-test', seed=SEED)

obs, info = test_envs.reset()  # Reset environment
a = test_envs.action_space.sample()  # Sample an action

obs, reward, terminate, truncate, info = test_envs.step(a)  # Step all environments with the sampled random actions
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
obs, reward, done, info = env.step(a)  # Step the environment with the sampled random action
assert (obs[-3:] == np.zeros(3)).all() # goal will be zeroed out because env is HiddenGoal

# You can choose to initialize the random seed of the environment.
# The state of your rng will remain unaffected after the environment is constructed.
env1 = door_open_goal_observable_cls(seed=5)
env2 = door_open_goal_observable_cls(seed=5)

env1.reset()  # Reset environment
env2.reset()
a1 = env1.action_space.sample()  # Sample an action
a2 = env2.action_space.sample()
next_obs1, _, _, _ = env1.step(a1)  # Step the environment with the sampled random action

next_obs2, _, _, _ = env2.step(a2)
assert (next_obs1[-3:] == next_obs2[-3:]).all() # 2 envs initialized with the same seed will have the same goal
assert not (next_obs2[-3:] == np.zeros(3)).all()   # The env's are goal observable, meaning the goal is not zero'd out

env3 = door_open_goal_observable_cls(seed=10)  # Construct an environment with a different seed
env1.reset()  # Reset environment
env3.reset()
a1 = env1.action_space.sample()  # Sample an action
a3 = env3.action_space.sample()
next_obs1, _, _, _ = env1.step(a1)  # Step the environment with the sampled random action
next_obs3, _, _, _ = env3.step(a3)

assert not (next_obs1[-3:] == next_obs3[-3:]).all() # 2 envs initialized with different seeds will have different goals
assert not (next_obs1[-3:] == np.zeros(3)).all()   # The env's are goal observable, meaning the goal is not zero'd out

```

## Citing Meta-World
In progress ... 

## Accompanying Baselines
In progress ... 

## Become a Contributor
We welcome all contributions to Meta-World. Please refer to the [contributor's guide](https://github.com/Farama-Foundation/Metaworld/blob/master/CONTRIBUTING.md) for how to prepare your contributions.

## Acknowledgements
Meta-World is now maintained by Farama-Foundation. You can interact with our community and Meta-World maintainers in our [Discord server](https://discord.gg/PfR7a79FpQ)

Meta-World is a work created by [Tianhe Yu (Stanford University)](https://cs.stanford.edu/~tianheyu/), [Deirdre Quillen (UC Berkeley)](https://scholar.google.com/citations?user=eDQsOFMAAAAJ&hl=en), [Zhanpeng He (Columbia University)](https://zhanpenghe.github.io), [Ryan Julian (University of Southern California)](https://ryanjulian.me), [Karol Hausman (Google AI)](https://karolhausman.github.io),  [Chelsea Finn (Stanford University)](https://ai.stanford.edu/~cbfinn/) and [Sergey Levine (UC Berkeley)](https://people.eecs.berkeley.edu/~svlevine/).

The code for Meta-World was originally based on [multiworld](https://github.com/vitchyr/multiworld), which is developed by [Vitchyr H. Pong](https://people.eecs.berkeley.edu/~vitchyr/), [Murtaza Dalal](https://github.com/mdalal2020), [Ashvin Nair](http://ashvin.me/), [Shikhar Bahl](https://shikharbahl.github.io), [Steven Lin](https://github.com/stevenlin1111), [Soroush Nasiriany](http://snasiriany.me/), [Kristian Hartikainen](https://hartikainen.github.io/) and [Coline Devin](https://github.com/cdevin). The Meta-World authors are grateful for their efforts on providing such a great framework as a foundation of our work. We also would like to thank Russell Mendonca for his work on reward functions for some of the environments.
