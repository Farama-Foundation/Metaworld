import numpy as np
import pytest

import metaworld
from metaworld.env_dict import ENV_NAMES

import gymnasium as gym

from tests.helpers import RandomMetaworldAgent


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_mt1_single_seed(env_name):
    agent = RandomMetaworldAgent(seed=42)
    env = gym.make("Meta-World/MT1",
                   env_name=env_name,
                   seed=42,
                   num_seeds_per_env=1)

    # TODO: rm duplicate code by introducing a helper function
    initial_observations = []
    agent.reset()
    obs, info = env.reset()
    initial_observations.append(obs)

    for _ in range(200):
        action = agent.get_action(obs, info, env.action_space)
        obs, reward, terminate, truncate, info = env.step(action)
        initial_observations.append(obs)
        if terminate or truncate:
            break

    second_observations = []
    agent.reset()
    obs, info = env.reset()
    second_observations.append(obs)
    for _ in range(200):
        action = agent.get_action(obs, info, env.action_space)
        obs, reward, terminate, truncate, info = env.step(action)
        second_observations.append(obs)
        if terminate or truncate:
            break

    assert np.array_equal(np.array(initial_observations), np.array(second_observations)), \
        f"Observations do not match for env {env_name} with single seed"
