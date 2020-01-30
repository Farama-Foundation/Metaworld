from metaworld.benchmarks.base import Benchmark
from metaworld.core.serializable import Serializable
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT


class ML1(MultiClassMultiTaskEnv, Benchmark, Serializable):

    def __init__(self, task_name, env_type='train', n_goals=50, sample_all=False):
        assert env_type == 'train' or env_type == 'test'
        Serializable.quick_init(self, locals())
        
        if task_name in HARD_MODE_CLS_DICT['train']:
            cls_dict = {task_name: HARD_MODE_CLS_DICT['train'][task_name]}
            args_kwargs = {task_name: HARD_MODE_ARGS_KWARGS['train'][task_name]}
        elif task_name in HARD_MODE_CLS_DICT['test']:
            cls_dict = {task_name: HARD_MODE_CLS_DICT['test'][task_name]}
            args_kwargs = {task_name: HARD_MODE_ARGS_KWARGS['test'][task_name]}
        else:
            raise NotImplementedError

        args_kwargs[task_name]['kwargs']['random_init'] = False
        super().__init__(
            task_env_cls_dict=cls_dict,
            task_args_kwargs=args_kwargs,
            sample_goals=True,
            obs_type='plain',
            sample_all=sample_all)

        goals = self.active_env.sample_goals_(n_goals)
        self.discretize_goal_space({task_name: goals})

    @classmethod
    def available_tasks(cls):
        key_train, key_test = HARD_MODE_ARGS_KWARGS['train'], HARD_MODE_ARGS_KWARGS['test']
        tasks = sum([list(key_train)], list(key_test))
        assert len(tasks) == 50
        return tasks

    @classmethod
    def get_train_tasks(cls, task_name, sample_all=False):
        return cls(task_name, env_type='train', n_goals=50, sample_all=sample_all)
    
    @classmethod
    def get_test_tasks(cls, task_name, sample_all=False):
        return cls(task_name, env_type='test', n_goals=10, sample_all=sample_all)
