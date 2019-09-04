# Metaworld
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/tianheyu927/metaworld/blob/master/LICENSE)

__Meta-World is an open-source simulated benchmark for meta-reinforcement learning and multi-task learning consisting of 50 distinct robotic manipulation tasks.__ We aim to provide task distributions that are sufficiently board to evaluate meta-RL algorithms' generalization ability to new behaviors.

For more information, please refer to our [website](corl2019metaworld.github.io).

## Installation
Meta-World is based on MuJoCo, which has a proprietary dependency we can't set up for you. Please follow the [instructions](https://github.com/openai/mujoco-py#install-mujoco) in the mujoco-py package for help. Once you're ready to install everything, clone this repository and install:

```
git clone https://github.com/tianheyu927/metaworld.git
cd metaworld
pip install -e .
```

# Using the benchmark
We provide benchmarks with a gradient of difficulties for meta-RL and multitask-RL:
* __ML1__ is a meta-RL benchmark environment for parameterized task distributions with options of 50 available task domains.
* __ML10__ is a meta-RL benchmark environment for non-parameterized task distributions with 10 task domains in train set and 5 task domains in the test set.
* __ML45__ is a meta-RL benchmark environment for non-parameterized task distributions with 45 task domains in train set and 5 task domains in the test set.
* __MT10__ is a multitask-RL benchmark environment for non-parameterized task distribution with 10 task domains.
* __MT50__ is a multitask-RL benchmark environment for non-parameterized task distribution with 50 task domains.


## Basics
We provide two extra API's to extend a [`gym.Env`](https://github.com/openai/gym/blob/c33cfd8b2cc8cac6c346bc2182cd568ef33b8821/gym/core.py#L8) interface for meta-RL and multitask-RL:
* sample_tasks(self, meta_batch_size): Return a list of tasks with a length of `meta_batch_size`.
* set_task(self, task): Set the task of a multitask environment.


## Using parameterized meta-RL benchmark ML1
Check out the available tasks:
```
from metaworld.benchmarks import ML1
print(ML1.available_tasks())
```
Create an environment with task `pick_place` for meta training:
```
env = ML1.get_train_tasks('pick_place')
```
or, alternatively:
```
env = ML1(task_name='pick_place', env_type='train')
```

## Using non-parameterized meta-RL benchmark ML10 and ML45
Create an environment with train tasks:
```
from metaworld.benchmarks import ML10
ml10_train_env = ML10.get_train_tasks()
```
or
```
ml10_train_env = ML10(env_type='train')
```
Create an environment with test tasks:
```
ml10_test_env = ML10.get_test_tasks()
```
or
```
ml10_test_env = ML10(env_type='test')
```

## Using non-parameterized multitask-RL benchmark MT10 and MT50
Create an environment with train tasks:
```
from metaworld.benchmarks import MT10
mt10_train_env = MT10.get_train_tasks()
```
or
```
mt10_train_env = MT10(env_type='train')
```
Create an environment with test tasks:
```
mt10_test_env = MT10.get_test_tasks()
```
or
```
mt10_test_env = MT10(env_type='test')
```

## Citing Meta-World
If you use Meta-World for your academic research, please kindly cite Meta-World with the following bibtex:

```
@misc{yu2018,
  Author = {Tianhe Yu and Deirdre Quillen and Zhanpeng He and Ryan Julian and Karol Hausman and Sergey Levine and Chelsea Finn},
  Title = {Meta-World: A Benchmark and Evaluation for Multi-Task and Meta- Reinforcement Learning},
  Year = {2019},
  url = "https://corl2019metaworld.github.io/"
}
```

## Acknowledgement
Meta-World is originally based on [multiworld](https://github.com/vitchyr/multiworld), which is developed by [Vitchyr H. Pong](https://people.eecs.berkeley.edu/~vitchyr/), Russell Mendonca and [Ashvin Nair](http://ashvin.me/). We would like to thank them for providing such a great framework as a foundation of our work.
