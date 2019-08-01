import pytest
import numpy as np

from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.sawyer_xyz.env_lists import HARD_MODE_LIST


@pytest.mark.parametrize('env_cls', HARD_MODE_LIST)
def test_single_env_multi_goals_discrete(env_cls):
    env_cls_dict = {'wrapped': env_cls}
    env_args_kwargs = {'wrapped': dict(args=[], kwargs={'obs_type': 'plain'})}
    multi_task_env = MultiClassMultiTaskEnv(
        task_env_cls_dict=env_cls_dict,
        task_args_kwargs=env_args_kwargs,
        sample_goals=True,
        obs_type='with_goal_idx'
    )
    goals = multi_task_env.active_env.sample_goals_(2)
    assert len(goals) == 2
    goals_dict = {'wrapped': goals}
    multi_task_env.discretize_goal_space(goals_dict)

    assert multi_task_env._fully_discretized
    tasks_with_goals = multi_task_env.sample_tasks(2)
    for t in tasks_with_goals:
        assert 'task' in t
        assert 'goal' in t
    multi_task_env.set_task(tasks_with_goals[0])
    assert multi_task_env._active_task == tasks_with_goals[0]['task']
    reset_obs = multi_task_env.reset()
    step_obs, _, _, _ = multi_task_env.step(multi_task_env.action_space.sample())
    assert np.all(multi_task_env.observation_space.shape == reset_obs.shape)
    assert np.all(multi_task_env.observation_space.shape == step_obs.shape)
    assert reset_obs[multi_task_env._max_plain_dim:][tasks_with_goals[0]['goal']] == 1
    assert step_obs[multi_task_env._max_plain_dim:][tasks_with_goals[0]['goal']] == 1
    assert np.sum(reset_obs[multi_task_env._max_plain_dim:]) == 1
    assert np.sum(reset_obs[multi_task_env._max_plain_dim:]) == 1


@pytest.mark.parametrize('env_list', [HARD_MODE_LIST[7:10], HARD_MODE_LIST[20:23]])
def test_multienv_multigoals_fully_discretized(env_list):
    env_cls_dict = {
        'env-{}'.format(i): env_cls
        for i, env_cls in enumerate(env_list)
    }
    env_args_kwargs = {
        'env-{}'.format(i): dict(args=[], kwargs={'obs_type': 'plain'})
        for i, _ in enumerate(env_list)
    }
    multi_task_env = MultiClassMultiTaskEnv(
        task_env_cls_dict=env_cls_dict,
        task_args_kwargs=env_args_kwargs,
        sample_goals=True,
        obs_type='with_goal_and_idx',
        sample_all=False,
    )
    goals_dict = dict()
    for i in range(len(multi_task_env._task_envs)):
        goals = multi_task_env.active_env.sample_goals_(i+1)
        goals_dict['env-{}'.format(i)] = goals
    multi_task_env.discretize_goal_space(goals_dict)
    assert multi_task_env._fully_discretized

    tasks_with_goals = multi_task_env.sample_tasks(2)
    for t in tasks_with_goals:
        assert 'task' in t
        assert 'goal' in t
    multi_task_env.set_task(tasks_with_goals[0])
    assert multi_task_env._active_task == tasks_with_goals[0]['task']

    # check task id
    reset_obs = multi_task_env.reset()
    step_obs, _, _, _ = multi_task_env.step(multi_task_env.action_space.sample())
    assert np.all(multi_task_env.observation_space.shape == reset_obs.shape)
    assert np.all(multi_task_env.observation_space.shape == step_obs.shape)

    task_name = multi_task_env._task_names[tasks_with_goals[0]['task']]
    goal = tasks_with_goals[0]['goal']
    plain_dim = multi_task_env._max_plain_dim
    task_start_index = multi_task_env._env_discrete_index[task_name]
    print('goal: {}, plain_dim: {}, task_start_index: {}'.format(goal, plain_dim, task_start_index))
    print(multi_task_env._env_discrete_index)
    assert reset_obs[plain_dim:][task_start_index + goal] == 1, reset_obs
    assert step_obs[plain_dim:][task_start_index + goal] == 1, step_obs
    assert np.sum(reset_obs[plain_dim: plain_dim + multi_task_env._n_discrete_goals]) == 1
    assert np.sum(reset_obs[plain_dim: plain_dim + multi_task_env._n_discrete_goals]) == 1


@pytest.mark.parametrize('env_list', [HARD_MODE_LIST[10:12], HARD_MODE_LIST[30:33]])
def test_multienv_single_goal(env_list):
    env_cls_dict = {
        'env-{}'.format(i): env_cls
        for i, env_cls in enumerate(env_list)
    }
    env_args_kwargs = {
        'env-{}'.format(i): dict(args=[], kwargs={'obs_type': 'plain'})
        for i, _ in enumerate(env_list)
    }
    multi_task_env = MultiClassMultiTaskEnv(
        task_env_cls_dict=env_cls_dict,
        task_args_kwargs=env_args_kwargs,
        sample_goals=False,
        obs_type='with_goal_and_idx',
        sample_all=True,
    )
    assert multi_task_env._fully_discretized

    n_tasks = len(env_list) * 2
    tasks = multi_task_env.sample_tasks(n_tasks)
    assert len(tasks) == n_tasks
    for t in tasks:
        multi_task_env.set_task(t)
        assert isinstance(multi_task_env.active_env,\
            env_cls_dict[multi_task_env._task_names[t % len(env_list)]])
