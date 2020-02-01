import gym
import memory_profiler
import pytest

from metaworld.benchmarks import ML45
from metaworld.envs.mujoco.sawyer_xyz.env_lists import HARD_MODE_LIST
from tests.helpers import step_env


def build_and_step(env_cls):
    env = env_cls()
    step_env(env, max_path_length=150, iterations=10, render=False)
    return env

def build_and_step_all(classes):
    envs = []
    for env_cls in classes:
        env = build_and_step(env_cls)
        envs += [env]

@pytest.fixture(scope='module')
def hard_mode_usage():
    profile = {}
    for env_cls in HARD_MODE_LIST:
        target = (build_and_step, [env_cls], {})
        memory_usage = memory_profiler.memory_usage(target)
        profile[env_cls] = max(memory_usage)

    return profile

@pytest.mark.parametrize('env_cls', HARD_MODE_LIST)
def test_max_memory_usage(env_cls, hard_mode_usage):
    # No env should use more  than 100MB
    #
    # Note: this is quite a bit higher than the average usage cap, because
    # loading a single environment incurs a fixed memory overhead which can't
    # be shared among environment in the same process
    assert hard_mode_usage[env_cls] < 250

def test_avg_memory_usage():
    # average usage no greater than 60MB/env
    target = (build_and_step_all, [HARD_MODE_LIST], {})
    usage = memory_profiler.memory_usage(target)
    average = max(usage) / len(HARD_MODE_LIST)
    assert average < 60

def test_from_task_memory_usage():
    target = (ML45.from_task, ['reach-v1'], {})
    usage = memory_profiler.memory_usage(target)
    assert max(usage) < 250
