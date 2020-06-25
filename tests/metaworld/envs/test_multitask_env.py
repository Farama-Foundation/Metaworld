import pytest
import numpy as np

from metaworld.benchmarks import ML10, ML45
from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT, MEDIUM_MODE_CLS_DICT, MEDIUM_MODE_ARGS_KWARGS, ALL_V1_ENVIRONMENTS
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerReachPushPickPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerReachPushPickPlaceWallEnv


HARD_MODE_LIST = (list(HARD_MODE_CLS_DICT['train'].values()) +
                  list(HARD_MODE_CLS_DICT['test'].values()))


@pytest.mark.parametrize('env_cls', ALL_V1_ENVIRONMENTS.values())
def test_single_env_multi_goals_discrete(env_cls):
    env_cls_dict = {'wrapped': env_cls}
    env_args_kwargs = {'wrapped': dict(args=[], kwargs={'task_id' : 1})}
    multi_task_env = MultiClassMultiTaskEnv(
        task_env_cls_dict=env_cls_dict,
        task_args_kwargs=env_args_kwargs,
        sample_goals=True,
        obs_type='with_goal_id'
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
    assert reset_obs[multi_task_env._max_obs_dim:][env_args_kwargs['wrapped']['kwargs']['task_id'] + tasks_with_goals[0]['goal']] == 1
    assert step_obs[multi_task_env._max_obs_dim:][env_args_kwargs['wrapped']['kwargs']['task_id'] + tasks_with_goals[0]['goal']] == 1
    assert np.sum(reset_obs[multi_task_env._max_plain_dim:]) == 1


@pytest.mark.parametrize('env_list', [HARD_MODE_LIST[7:10], HARD_MODE_LIST[20:23]])
def test_multienv_multigoals_fully_discretized(env_list):
    env_cls_dict = {
        'env-{}'.format(i): env_cls
        for i, env_cls in enumerate(env_list)
    }
    env_args_kwargs = {
        'env-{}'.format(i): dict(args=[], kwargs={'task_id' : i})
        for i, _ in enumerate(env_list)
    }
    multi_task_env = MultiClassMultiTaskEnv(
        task_env_cls_dict=env_cls_dict,
        task_args_kwargs=env_args_kwargs,
        sample_goals=True,
        obs_type='with_goal_id',
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
    plain_dim = multi_task_env._max_obs_dim
    task_start_index = multi_task_env.active_task
    # TODO these dims are ugly... rewrite assertion later
    assert reset_obs[plain_dim:][task_start_index] == 1, reset_obs
    assert step_obs[plain_dim:][task_start_index] == 1, step_obs
    assert np.sum(reset_obs[plain_dim + task_start_index: plain_dim + task_start_index + multi_task_env._n_discrete_goals]) == 1
    assert np.sum(reset_obs[plain_dim + task_start_index: plain_dim + task_start_index + multi_task_env._n_discrete_goals]) == 1

@pytest.mark.parametrize('env_list', [HARD_MODE_LIST[10:12], HARD_MODE_LIST[30:33]])
def test_multienv_single_goal(env_list):
    env_cls_dict = {
        'env-{}'.format(i): env_cls
        for i, env_cls in enumerate(env_list)
    }
    env_args_kwargs = {
        'env-{}'.format(i): dict(args=[], kwargs={'task_id' : i})
        for i, _ in enumerate(env_list)
    }
    multi_task_env = MultiClassMultiTaskEnv(
        task_env_cls_dict=env_cls_dict,
        task_args_kwargs=env_args_kwargs,
        sample_goals=False,
        obs_type='with_goal_id',
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


@pytest.mark.parametrize('env_cls',
    [SawyerReachPushPickPlaceEnv, SawyerReachPushPickPlaceWallEnv])
def test_reach_push_pick_place(env_cls):

    task_types = ['pick_place', 'reach', 'push']
    env_dict = {t: env_cls for t in task_types}
    env_args_kwargs = {
        t: dict(args=[], kwargs={'task_type': t, 'task_id' : 1})
        for t in task_types
    }

    multi_task_env = MultiClassMultiTaskEnv(
        task_env_cls_dict=env_dict,
        task_args_kwargs=env_args_kwargs,
        obs_type='with_goal_id',
        sample_goals=True,  # Each environment should still sample only
                            # one goal since each of them is discrete goal
                            # space and contains only one goal.
        sample_all=True,
    )
    goals_dict = {
        'pick_place': [np.array([0.1, 0.8, 0.2])],
        'reach': [np.array([-0.1, 0.8, 0.2])],
        'push': [np.array([0.1, 0.8, 0.02])],
    }
    multi_task_env.discretize_goal_space(goals_dict)
    assert multi_task_env._fully_discretized

    n_tasks = len(env_dict.keys())
    # do this test twice to make sure multiple sampling is working
    for _ in range(2):
        tasks = multi_task_env.sample_tasks(n_tasks)
        assert len(tasks) == n_tasks
        for t in tasks:
            assert 'task' in t.keys()
            assert 'goal' in t.keys()
            multi_task_env.set_task(t)
            _ = multi_task_env.reset()
            task_name = multi_task_env._task_names[t['task']]
            goal = multi_task_env.active_env.goal
            assert np.array_equal(goal, goals_dict[task_name][0])
            assert multi_task_env.active_env.task_type == task_name


# Full ml10 is too large for testing
ml3_env_cls_dict = {
    'pick-place-v1': MEDIUM_MODE_CLS_DICT['train']['pick-place-v1'],
    'reach-v1': MEDIUM_MODE_CLS_DICT['train']['reach-v1'],
    'sweep-v1': MEDIUM_MODE_CLS_DICT['train']['sweep-v1'],}
ml3_env_args_kwargs = {
    key: MEDIUM_MODE_ARGS_KWARGS['train'][key]
    for key in ml3_env_cls_dict.keys()
}
def test_ml3():
    multi_task_env = MultiClassMultiTaskEnv(
        task_env_cls_dict=ml3_env_cls_dict,
        task_args_kwargs=ml3_env_args_kwargs,
        sample_goals=True,
        obs_type='plain',
    )
    for _ in range(2):
        tasks = multi_task_env.sample_tasks(3)
        assert len(tasks) == 3
        for t in tasks:
            assert 'task' in t.keys()
            assert 'goal' in t.keys()
            multi_task_env.set_task(t)
            _ = multi_task_env.reset()
            goal = multi_task_env.active_env.goal
            assert multi_task_env.active_env.goal_space.contains(goal)


def test_task_name():
    task_names = MEDIUM_MODE_CLS_DICT['test'].keys()
    env = ML10.get_test_tasks()
    assert sorted(env.all_task_names) == sorted(task_names)

    _, _, _, info = env.step(env.action_space.sample())
    assert info['task_name'] in task_names

@pytest.mark.parametrize('observation_type', ['plain', 'with_goal_id'])
def test_observation_space(observation_type):
    env_cls_dict = {'pick-place-v1': MEDIUM_MODE_CLS_DICT['train']['pick-place-v1']}
    env_args_kwargs = {key: MEDIUM_MODE_ARGS_KWARGS['train'][key]
                       for key in env_cls_dict.keys()}
    multi_task_env = MultiClassMultiTaskEnv(
        task_env_cls_dict=env_cls_dict,
        task_args_kwargs=env_args_kwargs,
        sample_goals=True,
        obs_type=observation_type,
    )
    if observation_type == 'plain':
        assert multi_task_env.observation_space.shape == (9, )
    elif observation_type == 'with_goal_id':
        assert multi_task_env.observation_space.shape == (59, )


def test_action_space():
    env_cls_dict = {'pick-place-v1': MEDIUM_MODE_CLS_DICT['train']['pick-place-v1']}
    env_args_kwargs = {key: MEDIUM_MODE_ARGS_KWARGS['train'][key]
                       for key in env_cls_dict.keys()}
    multi_task_env = MultiClassMultiTaskEnv(
        task_env_cls_dict=env_cls_dict,
        task_args_kwargs=env_args_kwargs,
        sample_goals=True,
        obs_type="plain",
    )
    assert multi_task_env.action_space.shape == (4, )


@pytest.mark.parametrize('task_name', list(ALL_V1_ENVIRONMENTS.keys())[:10])
def test_static_task_ids(task_name):
    env = ML45.from_task(task_name)
    assert env.active_task == list(ALL_V1_ENVIRONMENTS.keys()).index(task_name)

