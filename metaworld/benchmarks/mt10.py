
from metaworld.benchmarks.base import Benchmark
from metaworld.core.serializable import Serializable
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import EASY_MODE_ARGS_KWARGS, EASY_MODE_CLS_DICT


class MT10(MultiClassMultiTaskEnv, Benchmark, Serializable):

    def __init__(self, env_type='train', sample_all=False):
        assert env_type == 'train' or env_type == 'test'
        Serializable.quick_init(self, locals())

        super().__init__(
            task_env_cls_dict=EASY_MODE_CLS_DICT,
            task_args_kwargs=EASY_MODE_ARGS_KWARGS,
            sample_goals=False,
            obs_type='with_goal_id',
            sample_all=sample_all,)

        goals_dict = {
            t: [e.goal.copy()]
            for t, e in zip(self._task_names, self._task_envs)
        }

        self.discretize_goal_space(goals_dict)
        assert self._fully_discretized
