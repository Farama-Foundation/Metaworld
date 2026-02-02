import os
import concurrent.futures

import numpy as np
import numpy.testing as npt

import metaworld

from tests.helpers import RandomMetaworldAgent, run_agent_episode


def _run_episode(seed: int, env_name: str) -> dict:
    agent = RandomMetaworldAgent(seed=seed)
    record_keys = ["observations"]
    ep_results = run_agent_episode(
        env_name=env_name,
        seed=seed,
        agent=agent,
        max_episode_steps=200,
        record_keys=record_keys,
    )

    return ep_results


def test_env_determinism_across_threads():
    """
    Test that running multiple episodes in parallel threads with the same seeds
    produces identical observations, ensuring thread safety and determinism.
    """

    max_workers = os.cpu_count() or 1

    num_parallel_eps = 55
    env_name = "reach-v3"

    num_batches = 3

    seeds = np.arange(num_parallel_eps)
    all_seeds = np.tile(seeds, num_batches)
    np.random.shuffle(all_seeds)

    all_futures = []
    batches = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for seed in all_seeds:
            fut = executor.submit(_run_episode, seed, env_name)
            batches.setdefault(seed, []).append(fut)
            all_futures.append(fut)

    for seed, batch_futures in batches.items():
        results = [f.result() for f in batch_futures]

        observations = [res["observations"] for res in results]

        try:
            for i in range(1, len(observations)):
                npt.assert_array_equal(
                    observations[0],
                    observations[i],
                    err_msg=f"Mismatch in seed {seed} between runs 0 and {i}"
                )
        except AssertionError as e:
            # Cancel all futures
            for f in all_futures:
                f.cancel()
            raise e
