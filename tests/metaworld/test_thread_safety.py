import os
import concurrent.futures

import gymnasium as gym
import numpy as np

import metaworld  # noqa: F401


def random_action(action_space, rng):
    low = action_space.low
    high = action_space.high
    return rng.uniform(low, high)


def _run_episode(seed):
    rng = np.random.default_rng(seed)
    env = gym.make('Meta-World/MT1',
                   env_name="reach-v3",
                   seed=seed,
                   disable_env_checker=True,
                   reward_function_version='v2',
                   max_episode_steps=None,
                   terminate_on_success=False,
                   num_goals=1,
                   )
    steps = 200
    observarions = []
    obs, _ = env.reset()
    observarions.append(obs)
    for _ in range(steps):
        action = random_action(env.action_space, rng)
        obs, _, done, truncated, _ = env.step(action)
        observarions.append(obs)
        if done or truncated:
            break
    return {"observations": np.array(observarions)}


def test_reach_v3_determinism_across_threads():
    max_workers = max(1, os.cpu_count() - 1)
    futures_1 = []
    futures_2 = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for seed in range(100):
            futures_1.append(executor.submit(_run_episode, seed))
            futures_2.append(executor.submit(_run_episode, seed))
    results_1 = [f.result() for f in futures_1]
    results_2 = [f.result() for f in futures_2]
    for i in range(100):
        obs_1 = results_1[i]["observations"]
        obs_2 = results_2[i]["observations"]
        assert np.array_equal(obs_1, obs_2), f"Mismatch in episode {i}"
