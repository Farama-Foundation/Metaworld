"""The public-facing Metaworld API."""

from __future__ import annotations

import abc
import pickle
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym  # type: ignore
import numpy as np
import numpy.typing as npt
from gymnasium import Env

# noqa: D104
from gymnasium.envs.registration import register
from gymnasium.spaces import Box, Space
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from gymnasium.vector.vector_env import VectorEnv
from gymnasium.wrappers.common import RecordEpisodeStatistics, TimeLimit
from numpy.typing import NDArray

import metaworld  # type: ignore
import metaworld.env_dict as _env_dict
from metaworld.env_dict import (
    ALL_V3_ENVIRONMENTS,
    ALL_V3_ENVIRONMENTS_GOAL_HIDDEN,
    ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE,
)
from metaworld.sawyer_xyz_env import SawyerXYZEnv  # type: ignore
from metaworld.types import Task


class MetaWorldEnv(abc.ABC):
    """Environment that requires a task before use.

    Takes no arguments to its constructor, and raises an exception if used
    before `set_task` is called.
    """

    @abc.abstractmethod
    def set_task(self, task: Task) -> None:
        """Sets the task.
        Args:
            task: The task to set.
        Raises:
            ValueError: If `task.env_name` is different from the current task.
        """
        raise NotImplementedError


class Benchmark(abc.ABC):
    """A Benchmark.

    When used to evaluate an algorithm, only a single instance should be used.
    """

    _train_classes: _env_dict.EnvDict
    _test_classes: _env_dict.EnvDict
    _train_tasks: list[Task]
    _test_tasks: list[Task]

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def train_classes(self) -> _env_dict.EnvDict:
        """Returns all of the environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> _env_dict.EnvDict:
        """Returns all of the environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> list[Task]:
        """Returns all of the training tasks for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> list[Task]:
        """Returns all of the test tasks for this benchmark."""
        return self._test_tasks


_ML_OVERRIDE = dict(partially_observable=True)
"""The overrides for the Meta-Learning benchmarks. Disables the inclusion of the goal position in the observation."""

_MT_OVERRIDE = dict(partially_observable=False)
"""The overrides for the Multi-Task benchmarks. Enables the inclusion of the goal position in the observation."""

_N_GOALS = 50
"""The number of goals to generate for each environment."""


def _encode_task(env_name, data) -> Task:
    """Instantiates a new `Task` object after pickling the data.

    Args:
        env_name: The name of the environment.
        data: The task data (will be pickled).

    Returns:
        A `Task` object.
    """
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(
    classes: _env_dict.EnvDict,
    args_kwargs: _env_dict.EnvArgsKwargsDict,
    kwargs_override: dict,
    seed: int | None = None,
) -> list[Task]:
    """Initialises goals for a given set of environments.

    Args:
        classes: The environment classes as an `EnvDict`.
        args_kwargs: The environment arguments and keyword arguments.
        kwargs_override: Any kwarg overrides.
        seed: The random seed to use.

    Returns:
        A flat list of `Task` objects, `_N_GOALS` for each environment in `classes`.
    """
    # Cache existing random state
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)

    tasks = []
    for env_name, args in args_kwargs.items():
        kwargs = args["kwargs"].copy()
        assert isinstance(kwargs, dict)
        assert len(args["args"]) == 0

        # Init env
        env = classes[env_name]()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs: list[npt.NDArray[Any]] = []

        # Set task
        del kwargs["task_id"]
        env._set_task_inner(**kwargs)

        for _ in range(_N_GOALS):  # Generate random goals
            env.reset()
            assert env._last_rand_vec is not None
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert (
            unique_task_rand_vecs.shape[0] == _N_GOALS
        ), f"Only generated {unique_task_rand_vecs.shape[0]} unique goals, not {_N_GOALS}"
        env.close()

        # Create a task for each random goal
        for rand_vec in rand_vecs:
            kwargs = args["kwargs"].copy()
            assert isinstance(kwargs, dict)
            del kwargs["task_id"]

            kwargs.update(dict(rand_vec=rand_vec, env_cls=classes[env_name]))
            kwargs.update(kwargs_override)

            tasks.append(_encode_task(env_name, kwargs))

        del env

    # Restore random state
    if seed is not None:
        np.random.set_state(st0)

    return tasks


# MT Benchmarks


class MT1(Benchmark):
    """The MT1 benchmark. A goal-conditioned RL environment for a single Metaworld task."""

    ENV_NAMES = list(_env_dict.ALL_V3_ENVIRONMENTS.keys())

    def __init__(self, env_name, seed=None):
        super().__init__()
        if env_name not in _env_dict.ALL_V3_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V3 environment")
        cls = _env_dict.ALL_V3_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(
            self._train_classes, {env_name: args_kwargs}, _MT_OVERRIDE, seed=seed
        )

        self._test_tasks = []


class MT10(Benchmark):
    """The MT10 benchmark. Contains 10 tasks in its train set. Has an empty test set."""

    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT10_V3
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT10_V3_ARGS_KWARGS
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed
        )

        self._test_tasks = []
        self._test_classes = []


class MT50(Benchmark):
    """The MT50 benchmark. Contains all (50) tasks in its train set. Has an empty test set."""

    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT50_V3
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT50_V3_ARGS_KWARGS
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed
        )

        self._test_tasks = []
        self._test_classes = []


# ML Benchmarks


class ML1(Benchmark):
    """The ML1 benchmark. A meta-RL environment for a single Metaworld task. The train and test set contain different goal positions.
    The goal position is not part of the observation."""

    ENV_NAMES = list(_env_dict.ALL_V3_ENVIRONMENTS.keys())

    def __init__(self, env_name, seed=None):
        super().__init__()
        if env_name not in _env_dict.ALL_V3_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V3 environment")

        cls = _env_dict.ALL_V3_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(
            self._train_classes, {env_name: args_kwargs}, _ML_OVERRIDE, seed=seed
        )
        self._test_tasks = _make_tasks(
            self._test_classes,
            {env_name: args_kwargs},
            _ML_OVERRIDE,
            seed=(seed + 1 if seed is not None else seed),
        )


class ML10(Benchmark):
    """The ML10 benchmark. Contains 10 tasks in its train set and 5 tasks in its test set. The goal position is not part of the observation."""

    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML10_V3["train"]
        self._test_classes = _env_dict.ML10_V3["test"]
        train_kwargs = _env_dict.ML10_ARGS_KWARGS["train"]

        test_kwargs = _env_dict.ML10_ARGS_KWARGS["test"]
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed
        )

        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, _ML_OVERRIDE, seed=seed
        )


class ML45(Benchmark):
    """The ML45 benchmark. Contains 45 tasks in its train set and 5 tasks in its test set (50 in total). The goal position is not part of the observation."""

    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML45_V3["train"]
        self._test_classes = _env_dict.ML45_V3["test"]
        train_kwargs = _env_dict.ML45_ARGS_KWARGS["train"]
        test_kwargs = _env_dict.ML45_ARGS_KWARGS["test"]

        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed
        )
        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, _ML_OVERRIDE, seed=seed
        )


class OneHotWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: Env, task_idx: int, num_tasks: int):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        env_lb = env.observation_space.low
        env_ub = env.observation_space.high
        one_hot_ub = np.ones(num_tasks)
        one_hot_lb = np.zeros(num_tasks)

        self.one_hot = np.zeros(num_tasks)
        self.one_hot[task_idx] = 1.0

        self._observation_space = gym.spaces.Box(
            np.concatenate([env_lb, one_hot_lb]), np.concatenate([env_ub, one_hot_ub])
        )

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    def observation(self, obs: NDArray) -> NDArray:
        return np.concatenate([obs, self.one_hot])


class RandomTaskSelectWrapper(gym.Wrapper):
    """A Gymnasium Wrapper to automatically set / reset the environment to a random
    task."""

    tasks: list[Task]
    sample_tasks_on_reset: bool = True

    def _set_random_task(self):
        task_idx = self.np_random.choice(len(self.tasks))
        self.unwrapped.set_task(self.tasks[task_idx])

    def __init__(
        self,
        env: Env,
        tasks: list[Task],
        sample_tasks_on_reset: bool = True,
        seed: int | None = None,
    ):
        super().__init__(env)
        self.tasks = tasks
        self.sample_tasks_on_reset = sample_tasks_on_reset
        if seed:
            self.unwrapped.seed(seed)

    def toggle_sample_tasks_on_reset(self, on: bool):
        self.sample_tasks_on_reset = on

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self.sample_tasks_on_reset:
            self._set_random_task()
        return self.env.reset(seed=seed, options=options)

    def sample_tasks(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ):
        self._set_random_task()
        return self.env.reset(seed=seed, options=options)


class PseudoRandomTaskSelectWrapper(gym.Wrapper):
    """A Gymnasium Wrapper to automatically reset the environment to a *pseudo*random task when explicitly called.

    Pseudorandom implies no collisions therefore the next task in the list will be used cyclically.
    However, the tasks will be shuffled every time the last task of the previous shuffle is reached.

    Doesn't sample new tasks on reset by default.
    """

    tasks: list[object]
    current_task_idx: int
    sample_tasks_on_reset: bool = False

    def _set_pseudo_random_task(self):
        self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
        if self.current_task_idx == 0:
            np.random.shuffle(self.tasks)
        self.unwrapped.set_task(self.tasks[self.current_task_idx])

    def toggle_sample_tasks_on_reset(self, on: bool):
        self.sample_tasks_on_reset = on

    def __init__(
        self,
        env: Env,
        tasks: list[object],
        sample_tasks_on_reset: bool = False,
        seed: int | None = None,
    ):
        super().__init__(env)
        self.sample_tasks_on_reset = sample_tasks_on_reset
        self.tasks = tasks
        self.current_task_idx = -1
        if seed:
            np.random.seed(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if self.sample_tasks_on_reset:
            self._set_pseudo_random_task()
        return self.env.reset(seed=seed, options=options)

    def sample_tasks(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ):
        self._set_pseudo_random_task()
        return self.env.reset(seed=seed, options=options)


class AutoTerminateOnSuccessWrapper(gym.Wrapper):
    """A Gymnasium Wrapper to automatically output a termination signal when the environment's task is solved.
    That is, when the 'success' key in the info dict is True.

    This is not the case by default in SawyerXYZEnv, because terminating on success during training leads to
    instability and poor evaluation performance. However, this behaviour is desired during said evaluation.
    Hence the existence of this wrapper.

    Best used *under* an AutoResetWrapper and RecordEpisodeStatistics and the like."""

    terminate_on_success: bool = True

    def __init__(self, env: Env):
        super().__init__(env)
        self.terminate_on_success = True

    def toggle_terminate_on_success(self, on: bool):
        self.terminate_on_success = on

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.terminate_on_success:
            terminated = info["success"] == 1.0
        return obs, reward, terminated, truncated, info


def _make_envs_common(
    benchmark,
    seed: int,
    max_episode_steps: int | None = None,
    use_one_hot: bool = True,
    terminate_on_success: bool = False,
) -> gym.vector.VectorEnv:
    if benchmark == "MT10":
        benchmark = MT10(seed=seed)

    def init_each_env(env_cls: type[SawyerXYZEnv], name: str, env_id: int) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        if terminate_on_success:
            env = AutoTerminateOnSuccessWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if use_one_hot:
            env = OneHotWrapper(env, env_id, len(benchmark.train_classes))
        tasks = [task for task in benchmark.train_tasks if task.env_name == name]
        env = RandomTaskSelectWrapper(env, tasks)
        env.action_space.seed(seed)
        return env

    return gym.vector.AsyncVectorEnv(
        [
            partial(init_each_env, env_cls=env_cls, name=name, env_id=env_id)
            for env_id, (name, env_cls) in enumerate(benchmark.train_classes.items())
        ]
    )


make_envs = partial(_make_envs_common, terminate_on_success=False)
make_eval_envs = partial(_make_envs_common, terminate_on_success=True)


def _make_single_env(
    name: str,
    seed: int = 0,
    max_episode_steps: int | None = None,
    use_one_hot: bool = False,
    env_id: int | None = None,
    num_tasks: int | None = None,
    terminate_on_success: bool = False,
) -> gym.Env:
    def init_each_env(env_cls: type[SawyerXYZEnv], name: str, seed: int) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        if terminate_on_success:
            env = AutoTerminateOnSuccessWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if use_one_hot:
            assert env_id is not None, "Need to pass env_id through constructor"
            assert num_tasks is not None, "Need to pass num_tasks through constructor"
            env = OneHotWrapper(env, env_id, num_tasks)

        if "test" in name:
            tasks = [task for task in benchmark.test_tasks if task.env_name in name]
        else:
            tasks = [task for task in benchmark.train_tasks if task.env_name in name]
        env = RandomTaskSelectWrapper(env, tasks, seed=seed)
        env.reset()
        return env

    if "MT1-" in name:
        name = name.replace("MT1-", "")
        benchmark = MT1(name, seed=seed)
        return init_each_env(
            env_cls=benchmark.train_classes[name], name=name, seed=seed
        )
    elif "ML1-" in name:
        benchmark = ML1(
            name.replace("ML1-train-" if "train" in name else "ML1-test-", ""),
            seed=seed,
        )  # type: ignore
        if "train" in name:
            env = init_each_env(
                env_cls=benchmark.train_classes[name.replace("ML1-train-", "")],
                name=name + "-train",
                seed=seed,
            )  # , init_each_env(env_cls=benchmark.test_classes[name.replace('ML1-', '')], name=name, seed=seed)
        elif "test" in name:
            env = init_each_env(
                env_cls=benchmark.test_classes[name.replace("ML1-test-", "")],
                name=name + "-test",
                seed=seed,
            )
        return env


make_single = partial(_make_single_env, terminate_on_success=False)


def register_mw_envs():
    for name in ALL_V3_ENVIRONMENTS:
        kwargs = {"name": "MT1-" + name}
        register(
            id=f"Meta-World/{name}", entry_point="metaworld:make_single", kwargs=kwargs
        )
        kwargs = {"name": "ML1-train-" + name}
        register(
            id=f"Meta-World/ML1-train-{name}",
            entry_point="metaworld:make_single",
            kwargs=kwargs,
        )
        kwargs = {"name": "ML1-test-" + name}
        register(
            id=f"Meta-World/ML1-test-{name}",
            entry_point="metaworld:make_single",
            kwargs=kwargs,
        )

    for name_hid in ALL_V3_ENVIRONMENTS_GOAL_HIDDEN:
        kwargs = {}
        register(
            id=f"Meta-World/{name_hid}",
            entry_point=lambda seed: ALL_V3_ENVIRONMENTS_GOAL_HIDDEN[name_hid](
                seed=seed
            ),
            kwargs=kwargs,
        )

    for name_obs in ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE:
        kwargs = {}
        register(
            id=f"Meta-World/{name_obs}",
            entry_point=lambda seed: ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[name_obs](
                seed=seed
            ),
            kwargs=kwargs,
        )

    kwargs = {}
    register(
        id="Meta-World/MT10-sync",
        vector_entry_point=lambda seed, use_one_hot, num_envs: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single,
                    "MT1-" + env_name,
                    num_tasks=10,
                    env_id=idx,
                    seed=None if not seed else seed + idx,
                    use_one_hot=use_one_hot,
                )
                for idx, env_name in enumerate(list(_env_dict.MT10_V3.keys()))
            ]
        ),
        kwargs=kwargs,
    )
    register(
        id="Meta-World/MT50-sync",
        vector_entry_point=lambda seed, use_one_hot, num_envs: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single,
                    "MT1-" + env_name,
                    num_tasks=50,
                    env_id=idx,
                    seed=None if not seed else seed + idx,
                    use_one_hot=use_one_hot,
                )
                for idx, env_name in enumerate(list(_env_dict.MT50_V3.keys()))
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/MT50-async",
        vector_entry_point=lambda seed, use_one_hot, num_envs: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single,
                    "MT1-" + env_name,
                    num_tasks=50,
                    env_id=idx,
                    seed=None if not seed else seed + idx,
                    use_one_hot=use_one_hot,
                )
                for idx, env_name in enumerate(list(_env_dict.MT50_V3.keys()))
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/ML10-train-sync",
        vector_entry_point=lambda seed, use_one_hot, num_envs: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single,
                    "ML1-train-" + env_name,
                    seed=None if not seed else seed + idx,
                    use_one_hot=False,
                )
                for idx, env_name in enumerate(list(_env_dict.ML10_V3["train"].keys()))
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/ML10-test-sync",
        vector_entry_point=lambda seed, use_one_hot, num_envs: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single,
                    "ML1-test-" + env_name,
                    seed=None if not seed else seed + idx,
                    use_one_hot=False,
                )
                for idx, env_name in enumerate(list(_env_dict.ML10_V3["test"].keys()))
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/ML10-train-async",
        vector_entry_point=lambda seed, use_one_hot, num_envs: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single,
                    "ML1-train-" + env_name,
                    seed=None if not seed else seed + idx,
                    use_one_hot=False,
                )
                for idx, env_name in enumerate(list(_env_dict.ML10_V3["train"].keys()))
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/ML10-test-async",
        vector_entry_point=lambda seed, use_one_hot, num_envs: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single,
                    "ML1-test-" + env_name,
                    seed=None if not seed else seed + idx,
                    use_one_hot=False,
                )
                for idx, env_name in enumerate(list(_env_dict.ML10_V3["test"].keys()))
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/MT10-async",
        vector_entry_point=lambda seed, use_one_hot, num_envs: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single,
                    "MT1-" + env_name,
                    num_tasks=10,
                    env_id=idx,
                    seed=None if not seed else seed + idx,
                    use_one_hot=use_one_hot,
                )
                for idx, env_name in enumerate(list(_env_dict.MT10_V3.keys()))
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/custom-mt-envs-sync",
        vector_entry_point=lambda seed, use_one_hot, envs_list, num_envs: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single,
                    "MT1-" + env_name,
                    num_tasks=len(envs_list),
                    env_id=idx,
                    seed=None if not seed else seed + idx,
                    use_one_hot=use_one_hot,
                )
                for idx, env_name in enumerate(envs_list)
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/custom-mt-envs-async",
        vector_entry_point=lambda seed, use_one_hot, envs_list, num_envs: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single,
                    "MT1-" + env_name,
                    num_tasks=len(envs_list),
                    env_id=idx,
                    seed=None if not seed else seed + idx,
                    use_one_hot=use_one_hot,
                )
                for idx, env_name in enumerate(envs_list)
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/custom-ml-envs-sync",
        vector_entry_point=lambda seed, use_one_hot, num_envs, envs_list: gym.vector.SyncVectorEnv(
            [
                partial(
                    make_single,
                    "ML1-train-" + env_name,
                    seed=None if not seed else seed + idx,
                    use_one_hot=False,
                )
                for idx, env_name in enumerate(envs_list)
            ]
        ),
        kwargs=kwargs,
    )

    register(
        id="Meta-World/custom-ml-envs-async",
        vector_entry_point=lambda seed, use_one_hot, num_envs, envs_list: gym.vector.AsyncVectorEnv(
            [
                partial(
                    make_single,
                    "ML1-train-" + env_name,
                    seed=None if not seed else seed + idx,
                    use_one_hot=False,
                )
                for idx, env_name in enumerate(envs_list)
            ]
        ),
        kwargs=kwargs,
    )


register_mw_envs()
__all__ = [
    "ML1",
    "MT1",
    "ML10",
    "MT10",
    "ML45",
    "MT50",
    "ALL_V3_ENVIRONMENTS_GOAL_HIDDEN",
    "ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE",
    "SawyerXYZEnv",
]
