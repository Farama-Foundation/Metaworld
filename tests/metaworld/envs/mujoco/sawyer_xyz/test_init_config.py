import pytest
import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.env_lists import HARD_MODE_LIST

# Note:
# I leave two environment's refactoring at the end since they
# contain more than one task.
@pytest.mark.parametrize('env_cls', HARD_MODE_LIST[7:])
def test_init_config(env_cls):
    env = env_cls()
    env.reset()
    assert 'init_config' in dir(env)
    assert np.all(env.goal_space.shape == env.goal.shape), 'goal: {}, goal_high: {}, goal_low: {}'.format(env.goal, env.goal_space.high, env.goal_space.low)
