import pytest

import metaworld
from metaworld.env_dict import ENV_NAMES

import gymnasium as gym

from tests.gym.helpers import ExpertPolicyMetaworldAgent, run_agent_episode_in_env


@pytest.mark.parametrize("env_name", ENV_NAMES)
# TODO: Add 'v1' back when all the reward functions are fixed to correctly report success.
@pytest.mark.parametrize("reward_function_version", ['v2'])
def test_policies(env_name, reward_function_version):
    agent = ExpertPolicyMetaworldAgent()

    max_episode_steps = 500
    num_episodes = 30

    env = gym.make("Meta-World/MT1",
                   env_name=env_name,
                   seed=42,
                   num_tasks_per_env=num_episodes,
                   max_episode_steps=max_episode_steps,
                   reward_function_version=reward_function_version,
                   task_sampler="pseudorandom",
                   )

    ep_seeds = []
    successes = 0
    for _ in range(num_episodes):
        ep_results = run_agent_episode_in_env(
            env=env,
            agent=agent,
            max_episode_steps=max_episode_steps,
        )
        ep_seeds.append(ep_results['env_seed'])
        successes += int(ep_results['agent_first_success_step'] is not None)

    env.close()

    success_rate = successes / num_episodes
    good_policy_success_rate = 0.8
    assert success_rate >= good_policy_success_rate, f"Success rate {success_rate} for env {env_name} below {good_policy_success_rate}"

    # Verify that all seeds are unique
    assert len(set(
        ep_seeds)) == num_episodes, f"Not all episode seeds are unique for env {env_name}"
