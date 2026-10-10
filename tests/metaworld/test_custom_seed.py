import gymnasium as gym
import numpy as np
import pytest

import metaworld  # noqa: F401


@pytest.mark.parametrize("vector_strategy", ["sync", "async"])
@pytest.mark.parametrize("seed", [0, 1])
def test_custom_mt_seed_matches_individual_environments(vector_strategy, seed):
    names = ["reach-v3", "push-v3"]
    envs = gym.make_vec(
        "Meta-World/custom-mt-envs",
        vector_strategy=vector_strategy,
        envs_list=names,
        seed=seed,
    )
    references = []
    try:
        for idx, name in enumerate(names):
            references.append(
                gym.make("Meta-World/MT1", env_name=name, seed=seed + idx)
            )

        for tasks, reference in zip(envs.get_attr("tasks"), references):
            assert tasks == reference.get_wrapper_attr("tasks")

        for _ in range(2):
            observations, _ = envs.reset()
            expected = np.stack([env.reset()[0] for env in references])
            np.testing.assert_array_equal(observations, expected)
            for _ in range(3):
                actions = np.stack([env.action_space.sample() for env in references])
                observations, rewards, terminated, truncated, _ = envs.step(actions)
                results = [env.step(action) for env, action in zip(references, actions)]
                np.testing.assert_array_equal(
                    observations, np.stack([r[0] for r in results])
                )
                np.testing.assert_array_equal(rewards, [r[1] for r in results])
                np.testing.assert_array_equal(terminated, [r[2] for r in results])
                np.testing.assert_array_equal(truncated, [r[3] for r in results])
    finally:
        envs.close()
        for env in references:
            env.close()
