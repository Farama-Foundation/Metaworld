import numpy as np
import pytest

import metaworld
import gymnasium as gym

from metaworld.env_dict import ENV_NAMES, MT_BENCHMARKS_TRAIN_ENV_NAMES


@pytest.mark.parametrize("mtx_benchmark_name", MT_BENCHMARKS_TRAIN_ENV_NAMES.keys())
def test_env_one_hot_wrapper(mtx_benchmark_name):
    """Test that the one-hot wrapper correctly encodes task information."""
    envs = gym.make_vec(f"Meta-World/{mtx_benchmark_name}", use_one_hot=True)
    obs, info = envs.reset()
    # 39:: one-hot part for MT10
    one_hots = []
    for i in range(envs.num_envs):
        one_hot_obs = obs[i][39:]
        # Check that only one index is 1 and the rest are 0s
        assert np.sum(one_hot_obs) == 1.0
        # Find the index of the 1
        one_index = np.argmax(one_hot_obs)
        one_hots.append(one_index)
    # Ensure that we have all tasks represented in the one-hot encodings
    assert set(one_hots) == set(range(envs.num_envs))
