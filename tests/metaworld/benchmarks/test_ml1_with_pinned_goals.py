import numpy as np
import pickle
import pytest

from metaworld.benchmarks import ML1WithPinnedGoal


@pytest.mark.parametrize('task_name', ML1WithPinnedGoal.available_tasks())
def test_ml1_with_pinned_goal(task_name):
    env = ML1WithPinnedGoal(task_name)
    num_variants = len(env._discrete_goals)
    tasks = env.sample_tasks(num_variants)

    def get_goals():
        goals = []
        for task in tasks:
            env.set_task(task)
            obs = env.reset()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            goals.append(info['goal'])
        return goals

    goals1 = get_goals()
    env = pickle.loads(pickle.dumps(env))
    goals2 = get_goals()

    assert np.array_equal(goals1, goals2)


