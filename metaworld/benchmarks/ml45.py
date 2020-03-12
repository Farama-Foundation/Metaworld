
from metaworld.benchmarks.base import Benchmark
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT


class ML45(MultiClassMultiTaskEnv, Benchmark):

    def __init__(self, env_type='train', sample_all=False, task_name=None):
        assert env_type == 'train' or env_type == 'test'

        if task_name is not None:
            if task_name not in HARD_MODE_CLS_DICT[env_type]:
                raise ValueError("{} does not exist in ML45 {} tasks".format(
                    task_name, env_type))
            cls_dict = {task_name: HARD_MODE_CLS_DICT[env_type][task_name]}
            args_kwargs = {task_name: HARD_MODE_ARGS_KWARGS[env_type][task_name]}
        else:
            cls_dict = HARD_MODE_CLS_DICT[env_type]
            args_kwargs = HARD_MODE_ARGS_KWARGS[env_type]

        super().__init__(
            task_env_cls_dict=cls_dict,
            task_args_kwargs=args_kwargs,
            sample_goals=True,
            obs_type='plain',
            sample_all=sample_all)

    @classmethod
    def from_task(cls, task_name):
        if task_name in HARD_MODE_CLS_DICT['train']:
            return cls(env_type='train', sample_all=True, task_name=task_name)
        elif task_name in HARD_MODE_CLS_DICT['test']:
            return cls(env_type='test', sample_all=True, task_name=task_name)
        else:
            raise ValueError('{} does not exist in ML45'.format(task_name))