# Meta-World
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)
[![Build Status](https://travis-ci.com/rlworkgroup/metaworld.svg?branch=master)](https://travis-ci.com/rlworkgroup/metaworld)

__Meta-World is an open-source simulated benchmark for meta-reinforcement learning and multi-task learning consisting of 50 distinct robotic manipulation tasks.__ We aim to provide task distributions that are sufficiently broad to evaluate meta-RL algorithms' generalization ability to new behaviors.

For more background information, please refer to our [website](https://meta-world.github.io) and the accompanying [conference publication](https://arxiv.org/abs/1910.10897), which **provides baseline results for 8 state-of-the-art meta- and multi-task RL algorithms**.

If you would like very infrequent announcements about the status of the benchmark, critical bugs and known issues before conference deadlines, and future plans, please join our mailing list: [metaworld-announce@googlegroups.com](https://groups.google.com/forum/#!forum/metaworld-announce).

__Table of Contents__
- [Installation](#installation)
- [Using the benchmark](#using-the-benchmark)
  * [Basics](#basics)
  * [Running ML1](#running-ml1)
  * [Running ML10 and ML45](#running-ml10-and-ml45)
  * [Running MT10 and MT50](#running-mt10-and-mt50)
  * [Running Single-Task Environments](#running-single-task-environments)
- [Citing Meta-World](#citing-meta-world)
- [Become a Contributor](#become-a-contributor)
- [Acknowledgements](#acknowledgements)

## Installation
Meta-World is based on MuJoCo, which has a proprietary dependency we can't set up for you. Please follow the [instructions](https://github.com/openai/mujoco-py#install-mujoco) in the mujoco-py package for help. Once you're ready to install everything, run:

```
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
```

Alternatively, you can clone the repository and install an editable version locally:

```
git clone https://github.com/rlworkgroup/metaworld.git
cd metaworld
pip install -e .
```

## Using the benchmark
Here is a list of benchmark environments for meta-RL (ML*) and multi-task-RL (MT*):
* [__ML1__](https://meta-world.github.io/figures/ml1.gif) is a meta-RL benchmark environment which tests few-shot adaptation to goal variation within single task. You can choose to test variation within any of [50 tasks](https://meta-world.github.io/figures/ml45-1080p.gif) for this benchmark.
* [__ML10__](https://meta-world.github.io/figures/ml10.gif) is a meta-RL benchmark which tests few-shot adaptation to new tasks. It comprises 10 meta-train tasks, and 3 test tasks.
* [__ML45__](https://meta-world.github.io/figures/ml45-1080p.gif) is a meta-RL benchmark which tests few-shot adaptation to new tasks. It comprises 45 meta-train tasks and 5 test tasks.
* [__MT10__](https://meta-world.github.io/figures/mt10.gif) and __MT50__ are multi-task-RL benchmark environments for learning a multi-task policy that perform 10 and 50 training tasks respectively. MT10 and MT50 augment environment observations with a one-hot vector which identifies the task.


### Basics
We provide two extra API's to extend a [`gym.Env`](https://github.com/openai/gym/blob/c33cfd8b2cc8cac6c346bc2182cd568ef33b8821/gym/core.py#L8) interface for meta-RL and multi-task-RL:
* `sample_tasks(self, meta_batch_size)`: Return a list of tasks with a length of `meta_batch_size`.
* `set_task(self, task)`: Set the task of a multi-task environment.


### Running ML1
```
from metaworld.benchmarks import ML1


print(ML1.available_tasks())  # Check out the available tasks

env = ML1.get_train_tasks('pick-place-v1')  # Create an environment with task `pick_place`
tasks = env.sample_tasks(1)  # Sample a task (in this case, a goal variation)
env.set_task(tasks[0])  # Set task

obs = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
```
### Running ML10 and ML45
Create an environment with train tasks:
```
from metaworld.benchmarks import ML10
ml10_train_env = ML10.get_train_tasks()
```
Create an environment with test tasks:
```
ml10_test_env = ML10.get_test_tasks()
```


### Running MT10 and MT50
Create an environment with train tasks:
```
from metaworld.benchmarks import MT10
mt10_train_env = MT10.get_train_tasks()
```

Create an environment with test tasks (noted that the train tasks and test tasks for multi-task (MT) environments are the same):
```
mt10_test_env = MT10.get_test_tasks()
```

### Running Single-Task Environments
Meta-World can also be used as a normal `gym.Env` for single task benchmarking. Here is an example of creating a `pick_place` environoment:
```
from metaworld.envs.mujoco.sawyer_xyz import SawyerReachPushPickPlaceEnv
env = SawyerReachPushPickPlaceEnv()
```

## Citing Meta-World
You use Meta-World for academic research, please kindly cite our CoRL 2019 paper the using following BibTeX entry.

```
@inproceedings{yu2019meta,
  title={Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning},
  author={Tianhe Yu and Deirdre Quillen and Zhanpeng He and Ryan Julian and Karol Hausman and Chelsea Finn and Sergey Levine},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2019}
  eprint={1910.10897},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
  url={https://arxiv.org/abs/1910.10897}
}
```

## Become a Contributor
We welcome all contributions to Meta-World. Please refer to the [contributor's guide](https://github.com/rlworkgroup/metaworld/blob/master/CONTRIBUTING.md) for how to prepare your contributions.

## Acknowledgements
Meta-World is a work by [Tianhe Yu (Stanford University)](https://cs.stanford.edu/~tianheyu/), [Deirdre Quillen (UC Berkeley)](https://scholar.google.com/citations?user=eDQsOFMAAAAJ&hl=en), [Zhanpeng He (Columbia University)](https://zhanpenghe.github.io), [Ryan Julian (University of Southern California)](https://ryanjulian.me), [Karol Hausman (Google AI)](https://karolhausman.github.io),  [Chelsea Finn (Stanford University)](https://ai.stanford.edu/~cbfinn/) and [Sergey Levine (UC Berkeley)](https://people.eecs.berkeley.edu/~svlevine/).

The code for Meta-World was originally based on [multiworld](https://github.com/vitchyr/multiworld), which is developed by [Vitchyr H. Pong](https://people.eecs.berkeley.edu/~vitchyr/), [Murtaza Dalal](https://github.com/mdalal2020), [Ashvin Nair](http://ashvin.me/), [Shikhar Bahl](https://shikharbahl.github.io), [Steven Lin](https://github.com/stevenlin1111), [Soroush Nasiriany](http://snasiriany.me/), [Kristian Hartikainen](https://hartikainen.github.io/) and [Coline Devin](https://github.com/cdevin). The Meta-World authors are grateful for their efforts on providing such a great framework as a foundation of our work. We also would like to thank Russell Mendonca for his work on reward functions for some of the environments.
