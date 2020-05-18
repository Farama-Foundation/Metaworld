
from gym.spaces import Box
from metaworld.benchmarks.base import Benchmark
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import EASY_MODE_ARGS_KWARGS, EASY_MODE_CLS_DICT
import numpy as np

class MT10(MultiClassMultiTaskEnv, Benchmark):

    def __init__(self, env_type="train", sample_all=False, task_name=None):
        del env_type

        if task_name is not None:
            if task_name not in EASY_MODE_CLS_DICT:
                raise ValueError("{} does not exist in MT10 tasks".format(
                    task_name))
            cls_dict = {task_name: EASY_MODE_CLS_DICT[task_name]}
            args_kwargs = {task_name: EASY_MODE_ARGS_KWARGS[task_name]}
        else:
            cls_dict = EASY_MODE_CLS_DICT
            args_kwargs = EASY_MODE_ARGS_KWARGS.copy()

        for env_args_kwargs in args_kwargs.values():
            env_args_kwargs['kwargs']['random_init'] = False

        super().__init__(
            task_env_cls_dict=cls_dict,
            task_args_kwargs=args_kwargs,
            sample_goals=False,
            obs_type='with_goal_id',
            sample_all=sample_all,)

        goals_dict = {
            t: [e.goal.copy()]
            for t, e in zip(self._task_names, self._task_envs)
        }

        self.discretize_goal_space(goals_dict)
        assert self._fully_discretized

    @classmethod
    def from_task(cls, task_name):
        if task_name in EASY_MODE_CLS_DICT:
            return cls(sample_all=True, task_name=task_name)
        else:
            raise ValueError('{} does not exist in MT10'.format(task_name))

    @property
    def observation_space(self):
        _max_obs_dim = 9
        _num_envs = 10
        obs_dim = _max_obs_dim + _num_envs
        return Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

    @property
    def active_task_one_hot(self):
        """Remove the last 40 element of obs's one hot because they are redundant
        (zeroed out due to their being 10 tasks in MT10 only).
        """
        one_hot = super().active_task_one_hot
        one_hot = one_hot[:-40]
        return one_hot
