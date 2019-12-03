import pytest

import numpy as np

from metaworld.benchmarks import ML1


@pytest.mark.parametrize('task_name', ML1.available_tasks())
def test_ml1_random_init(task_name):
    env = ML1.get_train_tasks(task_name)
    tasks = env.sample_tasks(1)
    env.set_task(tasks[0])

    actual_goal = env.active_env.goal

    _ = env.reset()
    _, _, _, info_0 = env.step(env.action_space.sample())
    
    _ = env.reset()
    _, _, _, info_1 = env.step(env.action_space.sample())

    assert np.all(info_0['goal'] == info_1['goal'])
    assert np.all(info_0['goal'] == actual_goal)
