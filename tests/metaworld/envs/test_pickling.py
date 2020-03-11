import numpy as np
import pickle as pkl
import pytest

from metaworld.benchmarks import ML1
from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT


@pytest.mark.parametrize('task', ML1.available_tasks())
def test_pickling(task):
    if task in HARD_MODE_CLS_DICT['train']:
        env = ML1.get_train_tasks(task)
    else:
        env = ML1.get_test_tasks(task)

    env2 = pkl.loads(pkl.dumps(env))

    assert len(env._task_names) == 1
    assert len(env._task_names) == len(env2._task_names)
    assert env._task_names[0] == env2._task_names[0]
    np.testing.assert_equal(env._discrete_goals, env2._discrete_goals)
