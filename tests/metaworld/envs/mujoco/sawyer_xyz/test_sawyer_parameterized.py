from gym.spaces import Discrete
import pytest
import numpy as np

from tests.helpers import step_env

from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS, _hard_mode_args_kwargs

@pytest.fixture(scope='module', params=list(ALL_V1_ENVIRONMENTS.keys()))
def env(request):
    env_cls = ALL_V1_ENVIRONMENTS[request.param]
    env_args_kwargs = _hard_mode_args_kwargs(env_cls, request.param)
    env_args = env_args_kwargs['args']
    env_kwargs = env_args_kwargs['kwargs']
    del env_kwargs['task_id']
    env = env_cls(*env_args, **env_kwargs)

    yield env

    # clean-up
    env.close()

def test_all_envs_step(env):
    step_env(env, max_path_length=10)

def test_obs_type(env):
    o = env.reset()
    o_g = env._get_obs()
    space = env.observation_space
    assert space.shape == o.shape, 'type: {}, env: {}'.format(t, env)
    assert space.shape == o_g.shape, 'type: {}, env: {}'.format(t, env)

def test_discretize_goal_space(env):
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

# Note:
# I leave two environment's refactoring at the end since they
# contain more than one task.
def test_init_config(env):
    env.reset()
    assert 'init_config' in dir(env)
    assert np.all(env.goal_space.shape == env.goal.shape), 'goal: {}, goal_high: {}, goal_low: {}'.format(env.goal, env.goal_space.high, env.goal_space.low)
