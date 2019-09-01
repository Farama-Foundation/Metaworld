
from metaworld.benchmarks.base import Benchmark
from metaworld.core.serializable import Serializable
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT


class MT50(MultiClassMultiTaskEnv, Benchmark):

    def __init__(self, env_type='train', sample_all=False):
        assert env_type == 'train' or env_type == 'test'
        Serializable.quick_init(self, locals())

        cls_dict = {}
        args_kwargs = {}
        for k in HARD_MODE_CLS_DICT.keys():
            for i in HARD_MODE_CLS_DICT[k].keys():
                key = '{}-{}'.format(k, i)
                cls_dict[key] = HARD_MODE_CLS_DICT[k][i]
                args_kwargs[key] = HARD_MODE_ARGS_KWARGS[k][i]
        assert len(cls_dict.keys()) == 50

        super().__init__(
            task_env_cls_dict=cls_dict,
            task_args_kwargs=args_kwargs,
            sample_goals=False,
            obs_type='with_goal_idx',
            sample_all=sample_all,)

        goals_dict = {
            t: [e.goal.copy()]
            for t, e in zip(self._task_names, self._task_envs)
        }

        self.discretize_goal_space(goals_dict)
        assert self._fully_discretized
