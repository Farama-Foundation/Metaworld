import numpy as np
import pytest

from metaworld.env_dict import ENV_NAMES
from tests.gym.helpers import RandomMetaworldAgent, run_agent_episode


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_env_seeds_produce_unique_observations(env_name):
    agent = RandomMetaworldAgent(seed=42)

    # The initial observation is sufficient to verify different seeds produce different results
    # since it contains randomized object positions and the goal position.
    max_episode_steps = 1

    record_keys = set(['observations'])
    seeds = [42, 43, 44, 45, 46]
    observations = []

    for seed in seeds:
        ep_results = run_agent_episode(
            env_name=env_name,
            seed=seed,
            agent=agent,
            max_episode_steps=max_episode_steps,
            record_keys=record_keys,
        )
        observations.append(ep_results['observations'][0])

    # Verify that all observations are unique
    unique_observations = {tuple(obs) for obs in observations}
    assert len(unique_observations) == len(seeds), \
        f"Not all observations are unique for env {env_name} with different seeds"
