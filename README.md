# Metaworld
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/tianheyu927/metaworld/blob/master/LICENSE)


Metaworld is a multitask manipulation environment that is currently under development. If you’d like to contribute to creating new environments or running algorithms on these environments, please let us know so we can coordinate efforts.

# Installation
Metaworld is based on MuJoCo, which has a proprietary dependency we can't set up for you. Please follow the [instructions](https://github.com/openai/mujoco-py#install-mujoco) in the mujoco-py package for help. Once you're ready to install everything, clone this repository and install:

```
git clone https://github.com/tianheyu927/metaworld.git
cd metaworld
pip install -e .
```

# Using the benchmark
We provide benchmarks with a gradient of difficulties for meta-RL and multitask-RL:
* ML1 is a meta-RL benchmark environment for parameterized task distributions with options of 50 available task domains.
* ML10 is a meta-RL benchmark environment for non-parameterized task distributions with 10 task domains in train set and 5 task domains in the test set.
* ML45 is a meta-RL benchmark environment for non-parameterized task distributions with 45 task domains in train set and 5 task domains in the test set.
* MT10 is a multitask-RL benchmark environment for non-parameterized task distribution with 10 task domains.
* MT50 is a multitask-RL benchmark environment for non-parameterized task distribution with 50 task domains.


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
