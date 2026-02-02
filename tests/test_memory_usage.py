import multiprocessing as mp

import memory_profiler
import pytest
import gymnasium as gym

from metaworld.env_dict import ALL_V3_ENVIRONMENTS

from tests.helpers import run_agent_episode_in_env, RandomMetaworldAgent

mp.set_start_method('spawn', force=True)


def _build_env_and_run_eps(env_name):
    seed = 42
    agent = RandomMetaworldAgent(seed=seed)
    max_episode_steps = 150
    env = gym.make("Meta-World/MT1",
                   env_name=env_name,
                   seed=seed,
                   num_tasks_per_env=1,
                   max_episode_steps=max_episode_steps,)

    episodes = 10
    for _ in range(episodes):
        run_agent_episode_in_env(
            env=env,
            agent=agent,
            max_episode_steps=max_episode_steps,
        )


@pytest.mark.parametrize("env_name", ALL_V3_ENVIRONMENTS.keys())
def test_env_memory_profiler(env_name):
    target = (_build_env_and_run_eps, [env_name], {})
    memory_usage = memory_profiler.memory_usage(target)

    # Max memory usage per env in MB
    env_max_memory_usage_threshold = 300
    print(f"Memory usage for env {env_name}: {memory_usage}")
    env_max_memory_usage = max(memory_usage)
    assert env_max_memory_usage < env_max_memory_usage_threshold, f"Env {env_name} exceeded max memory usage of {env_max_memory_usage_threshold}: {env_max_memory_usage}MB"
