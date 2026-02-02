import os
import concurrent.futures

import numpy as np

import metaworld

from tests.helpers import RandomMetaworldAgent, run_agent_episode


def _run_episode(seed):
    agent = RandomMetaworldAgent(seed=seed)
    record_keys = ["observations"]
    ep_results = run_agent_episode(
        env_name="reach-v3",
        seed=seed,
        agent=agent,
        max_episode_steps=200,
        record_keys=record_keys,
    )

    return ep_results


def test_determinism_across_threads():
    """
    Test that running multiple episodes in parallel threads with the same seeds
    produces identical observations, ensuring thread safety and determinism.
    """

    max_workers = max(1, os.cpu_count() - 1)

    num_parallel_eps = 100

    futures_1 = []
    futures_2 = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for seed in range(num_parallel_eps):
            futures_1.append(executor.submit(_run_episode, seed))
            futures_2.append(executor.submit(_run_episode, seed))
    results_1 = [f.result() for f in futures_1]
    results_2 = [f.result() for f in futures_2]
    for i in range(num_parallel_eps):
        obs_1 = results_1[i]["observations"]
        obs_2 = results_2[i]["observations"]
        assert np.array_equal(obs_1, obs_2), f"Mismatch in episode {i}"
