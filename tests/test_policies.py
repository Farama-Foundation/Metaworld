import numpy as np
import pytest

import metaworld
from metaworld.env_dict import ENV_NAMES

import gymnasium as gym

from tests.helpers import ExpertPolicyMetaworldAgent, run_agent_episode


@pytest.mark.parametrize("env_name", ENV_NAMES)
# TODO: Add 'v1' back when all the reward functions are fixed to correctly report success.
@pytest.mark.parametrize("reward_function_version", ['v2'])
def test_policies(env_name, reward_function_version):
    agent = ExpertPolicyMetaworldAgent()

    max_episode_steps = 500
    num_episodes = 30

    rng = np.random.default_rng(42)

    successes = 0
    for _ in range(num_episodes):
        ep_seed = rng.integers(0, 1_000_000)
        ep_results = run_agent_episode(
            env_name=env_name,
            seed=ep_seed,
            agent=agent,
            max_episode_steps=max_episode_steps,
            reward_function_version=reward_function_version,
        )
        successes += int(ep_results['agent_first_success_step'] is not None)

    success_rate = successes / num_episodes
    good_policy_success_rate = 0.8
    assert success_rate >= good_policy_success_rate, f"Success rate {success_rate} for env {env_name} below {good_policy_success_rate}"
