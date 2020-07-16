"""Proposal for a simple, understandable MetaWorld API."""
import abc
import pickle
from collections import OrderedDict
from typing import List, NamedTuple, Type

import metaworld.envs.mujoco.env_dict as _env_dict
import numpy as np


EnvName = str


class Task(NamedTuple):
    """All data necessary to describe a single MDP.

    Should be passed into a MetaWorldEnv's set_task method.
    """

    env_name: EnvName
    data: bytes  # Contains env parameters like random_init and *a* goal


class MetaWorldEnv:
    """Environment that requires a task before use.

    Takes no arguments to its constructor, and raises an exception if used
    before `set_task` is called.
    """

    def set_task(self, task: Task) -> None:
        """Set the task.

        Raises:
            ValueError: If task.env_name is different from the current task.

        """


class Benchmark(abc.ABC):
    """A Benchmark.

    When used to evaluate an algorithm, only a single instance should be used.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def train_classes(self) -> 'OrderedDict[EnvName, Type]':
        """Get all of the environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> 'OrderedDict[EnvName, Type]':
        """Get all of the environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> List[Task]:
        """Get all of the training tasks for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> List[Task]:
        """Get all of the test tasks for this benchmark."""
        return self._test_tasks


_ML_OVERRIDE = dict(partially_observable=True)
_MT_OVERRIDE = dict(partially_observable=False)

_N_GOALS = 50


def _encode_task(env_name, data):
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(classes, args_kwargs, kwargs_override):
    tasks = []
    for (env_name, args) in args_kwargs.items():
        assert len(args['args']) == 0
        env_cls = classes[env_name]
        env = env_cls()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs = []
        kwargs = args['kwargs'].copy()
        del kwargs['task_id']
        env._set_task_inner(**kwargs)
        for _ in range(_N_GOALS):
            env.reset()
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert unique_task_rand_vecs.shape[0] == _N_GOALS

        env.close()
        for rand_vec in rand_vecs:
            kwargs = args['kwargs'].copy()
            del kwargs['task_id']
            kwargs.update(dict(rand_vec=rand_vec, env_cls=env_cls))
            kwargs.update(kwargs_override)
            tasks.append(_encode_task(env_name, kwargs))
    return tasks


def _ml1_env_names():
    key_train = _env_dict.HARD_MODE_ARGS_KWARGS['train']
    key_test = _env_dict.HARD_MODE_ARGS_KWARGS['test']
    tasks = sum([list(key_train)], list(key_test))
    assert len(tasks) == 50
    return tasks


class ML1(Benchmark):

    ENV_NAMES = _ml1_env_names()

    def __init__(self, env_name):
        super().__init__()
        try:
            cls = _env_dict.HARD_MODE_CLS_DICT['train'][env_name]
            args_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['train'][env_name]
        except KeyError:
            cls = _env_dict.HARD_MODE_CLS_DICT['test'][env_name]
            args_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['test'][env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        self._train_ = OrderedDict([(env_name, cls)])
        self._train_tasks = _make_tasks(self._train_classes,
                                        {env_name: args_kwargs},
                                        _ML_OVERRIDE)
        self._test_tasks = _make_tasks(self._test_classes,
                                        {env_name: args_kwargs},
                                        _ML_OVERRIDE)
        
class MT1(Benchmark):

    ENV_NAMES = _ml1_env_names()

    def __init__(self, env_name):
        super().__init__()
        try:
            cls = _env_dict.HARD_MODE_CLS_DICT['train'][env_name]
            args_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['train'][env_name]
        except KeyError:
            cls = _env_dict.HARD_MODE_CLS_DICT['test'][env_name]
            args_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['test'][env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = OrderedDict()
        self._train_ = OrderedDict([(env_name, cls)])
        self._train_tasks = _make_tasks(self._train_classes,
                                        {env_name: args_kwargs},
                                        _MT_OVERRIDE)
        self._test_tasks = []


class ML10(Benchmark):

    def __init__(self):
        super().__init__()
        self._train_classes = _env_dict.MEDIUM_MODE_CLS_DICT['train']
        self._test_classes = _env_dict.MEDIUM_MODE_CLS_DICT['test']
        train_kwargs = _env_dict.medium_mode_train_args_kwargs
        self._train_tasks = _make_tasks(self._train_classes,
                                        train_kwargs,
                                        _ML_OVERRIDE)
        test_kwargs = _env_dict.medium_mode_test_args_kwargs
        self._test_tasks = _make_tasks(self._test_classes,
                                       test_kwargs,
                                       _ML_OVERRIDE)


class ML45(Benchmark):

    def __init__(self):
        super().__init__()
        self._train_classes = _env_dict.HARD_MODE_CLS_DICT['train']
        self._test_classes = _env_dict.HARD_MODE_CLS_DICT['test']
        train_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['train']
        self._train_tasks = _make_tasks(self._train_classes,
                                        train_kwargs,
                                        _ML_OVERRIDE)
        self._test_tasks = _make_tasks(self._test_classes,
                                       _env_dict.HARD_MODE_ARGS_KWARGS['test'],
                                       _ML_OVERRIDE)


class MT10(Benchmark):

    def __init__(self):
        super().__init__()
        self._train_classes = _env_dict.EASY_MODE_CLS_DICT
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.EASY_MODE_ARGS_KWARGS
        self._train_tasks = _make_tasks(self._train_classes,
                                        train_kwargs,
                                        _MT_OVERRIDE)
        self._test_tasks = []


class MT50(Benchmark):

    def __init__(self):
        super().__init__()
        self._train_classes = _env_dict.HARD_MODE_CLS_DICT['train'].copy()
        # We're going to modify it, so make a copy
        train_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['train'].copy()
        test_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['test']
        for (env_name, cls) in _env_dict.HARD_MODE_CLS_DICT['test'].items():
            assert env_name not in self._train_classes
            assert env_name not in train_kwargs
            self._train_classes[env_name] = cls
            train_kwargs[env_name] = test_kwargs[env_name]
        self._test_classes = OrderedDict()
        self._train_tasks = _make_tasks(self._train_classes,
                                        train_kwargs,
                                        _MT_OVERRIDE)
        self._test_tasks = []
