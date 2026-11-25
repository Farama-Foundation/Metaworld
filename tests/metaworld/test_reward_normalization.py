import gymnasium as gym
import numpy as np
import pytest

from metaworld.wrappers import NormalizeRewardsExponential


class RewardSequenceEnv(gym.Env):
    def reset(self, *, seed=None, options=None):
        self.rewards = iter([2.0, 4.0])
        return np.zeros(1), {}

    def step(self, action):
        return np.zeros(1), next(self.rewards), False, False, {}


def test_exponential_reward_normalization_counts_each_reward_once():
    env = NormalizeRewardsExponential(0.5, RewardSequenceEnv())
    try:
        env.reset()
        _, reward, terminated, truncated, info = env.step(None)
        assert env._reward_mean == pytest.approx(1.0)
        assert env._reward_var == pytest.approx(1.0)
        assert reward == pytest.approx(2.0 / (1.0 + 1e-8))
        assert not terminated and not truncated and info == {}

        _, reward, _, _, _ = env.step(None)
        assert env._reward_mean == pytest.approx(2.5)
        assert env._reward_var == pytest.approx(1.625)
        assert reward == pytest.approx(4.0 / (np.sqrt(1.625) + 1e-8))

        # Running statistics continue across episode boundaries.
        env.reset()
        _, reward, _, _, _ = env.step(None)
        assert env._reward_mean == pytest.approx(2.25)
        assert env._reward_var == pytest.approx(0.84375)
        assert reward == pytest.approx(2.0 / (np.sqrt(0.84375) + 1e-8))
    finally:
        env.close()
