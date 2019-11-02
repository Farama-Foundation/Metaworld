#!/usr/bin/env python3
"""Test script for profiling average memory footprint."""
import pprint

import memory_profiler

from metaworld.envs.mujoco.sawyer_xyz.env_lists import HARD_MODE_LIST
from tests.helpers import step_env


def build_and_step(env_cls):
    env = env_cls()
    env.reset()
    step_env(env, max_path_length=1000, iterations=10)
    return env

def build_and_step_all(classes):
    envs = []
    for env_cls in classes:
        env = build_and_step(env_cls)
        envs += [env]

def profile_hard_mode_indepedent():
    profile = {}
    for env_cls in HARD_MODE_LIST:
        target = (build_and_step, [env_cls], {})
        memory_usage = memory_profiler.memory_usage(target)
        profile[env_cls] = max(memory_usage)

    return profile

def profile_hard_mode_shared():
    target = (build_and_step_all, [HARD_MODE_LIST], {})
    usage = memory_profiler.memory_usage(target)
    return max(usage)


if __name__ == '__main__':
    profile = profile_hard_mode_indepedent()
    print('--------- Independent memory footprints ---------')
    for cls, u in profile.items():
        print('{:<40} {:>5.1f} MB'.format(cls.__name__, u))
    max_independent = max(profile.values())
    mean_independent = sum(profile.values()) / len(profile)
    min_independent = min(profile.values())
    print('\nSummary:')
    print('| min      | mean     | max      |')
    print('|----------|----------|----------|')
    print('| {:.1f} MB | {:.1f} MB | {:.1f} MB |'
          .format(min_independent, mean_independent, max_independent))
    print('\n')

    print('---------    Shared memory footprint    ---------')
    max_usage = profile_hard_mode_shared()
    mean_shared = max_usage / len(HARD_MODE_LIST)
    print('Mean memory footprint (n = {}): {:.1f} MB'
          .format(len(HARD_MODE_LIST), mean_shared))