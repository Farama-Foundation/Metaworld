import numpy as np
import pytest

import metaworld
from metaworld.env_dict import ENV_NAMES

import gymnasium as gym

from tests.helpers import RandomMetaworldAgent, run_agent_episode, run_agent_episode_in_env


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_mt1_single_seed(env_name):
    agent = RandomMetaworldAgent(seed=42)

    max_episode_steps = 200
    record_keys = set(['observations'])
    seed = 42

    env = gym.make("Meta-World/MT1",
                   env_name=env_name,
                   seed=seed,
                   num_seeds_per_env=1,
                   max_episode_steps=max_episode_steps,)

    first_ep = run_agent_episode_in_env(
        env=env,
        agent=agent,
        max_episode_steps=max_episode_steps,
        record_keys=record_keys,
    )
    # Verify that for the num_seeds_per_env=1 and MT1 env the seed is the same as the one passed
    assert first_ep['env_seed'] == seed, f"Env seed {first_ep['env_seed']} does not match passed seed {seed}"

    # Reset the env and run everything again
    second_ep = run_agent_episode_in_env(
        env=env,
        agent=agent,
        max_episode_steps=max_episode_steps,
        record_keys=record_keys,
    )
    # Verify that the seed is still the same
    assert second_ep['env_seed'] == seed, f"Env seed {second_ep['env_seed']} does not match passed seed {seed}"

    # Verify that the observations are the same across both runs
    obs_first = first_ep['observations']
    obs_second = second_ep['observations']
    assert np.array_equal(np.array(obs_first), np.array(obs_second)), \
        f"Observations do not match for env {env_name} with single seed"

    # --- Do another run with a new env to verify that it also matches ---
    third_ep = run_agent_episode(
        env_name=env_name,
        agent=agent,
        max_episode_steps=max_episode_steps,
        record_keys=record_keys,
        seed=seed,
    )
    # Verify that the seed is still the same
    assert third_ep['env_seed'] == seed, f"Env seed {third_ep['env_seed']} does not match passed seed {seed}"

    obs_third = third_ep['observations']
    assert np.array_equal(np.array(obs_first), np.array(obs_third)), \
        f"Observations do not match for env {env_name} with single seed"
