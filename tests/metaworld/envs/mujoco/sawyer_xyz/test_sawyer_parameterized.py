from gym.spaces import Discrete
import pytest
import numpy as np

from tests.helpers import step_env

from metaworld.envs.mujoco.sawyer_xyz.base import OBS_TYPE
from metaworld.envs.mujoco.sawyer_xyz.env_lists import HARD_MODE_LIST


@pytest.mark.parametrize('env_cls', HARD_MODE_LIST)
def test_sawyer(env_cls):
    env = env_cls()
    step_env(env, max_path_length=10)
    env.close()


@pytest.mark.parametrize('env_cls', HARD_MODE_LIST)
def test_obs_type(env_cls):
    for t in OBS_TYPE:
        if t == 'with_goal' or t == 'plain':
            env = env_cls(obs_type=t)
            o = env.reset()
            o_g = env._get_obs()
            space = env.observation_space
            assert space.shape == o.shape, 'type: {}, env: {}'.format(t, env)
            assert space.shape == o_g.shape, 'type: {}, env: {}'.format(t, env)
            env.close()


@pytest.mark.parametrize('env_cls', HARD_MODE_LIST)
def test_discretize_goal_space(env_cls):
    env = env_cls()
    discrete_goals = env.sample_goals_(2)
    env.discretize_goal_space(goals=discrete_goals)
    assert isinstance(env.discrete_goal_space, Discrete)
    assert env.discrete_goal_space.n == 2

    # test discrete sampling and setting
    goals = env.sample_goals_(2)
    for g in goals:
        env.set_goal_(g)
        assert g in env.discrete_goal_space
        assert np.all(env.goal == discrete_goals[g])

    env.close()
