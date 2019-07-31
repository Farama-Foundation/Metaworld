from gym.spaces import Discrete
import numpy as np
import pytest

from metaworld.envs.mujoco.sawyer_xyz.env_lists import HARD_MODE_LIST


@pytest.mark.parametrize('env_cls', HARD_MODE_LIST)
def test_discretize_goal_space(env_cls):
    env = env_cls()
    discrete_goals = env.sample_goals_(2)
    env.discretize_goal_space(goals=discrete_goals)
    assert env.discrete_goal_space == True
    assert isinstance(env.goal_space, Discrete)
    assert env.goal_space.n == 2

    # test discrete sampling and setting
    goals = env.sample_goals_(2)
    for g in goals:
        env.set_goal_(g)
        assert g in env.goal_space
        assert np.all(env.goal == discrete_goals[g])
