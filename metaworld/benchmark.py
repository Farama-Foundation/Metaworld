from typing import Any, NamedTuple, Sequence
from typing_extensions import Literal

import copy

import numpy as np

from metaworld.utils.numpy import randint
from metaworld.env_dict import MT_BENCHMARKS_TRAIN_ENV_NAMES, ML_BENCHMARKS

_ML_ENV_KWARGS_OVERRIDE = dict(goal_observable=False)
"""The overrides for the Meta-Learning benchmarks. Disables the inclusion of the goal position in the observation."""

_MT_ENV_KWARGS_OVERRIDE = dict(goal_observable=True)
"""The overrides for the Multi-Task benchmarks. Enables the inclusion of the goal position in the observation."""

_DEFAULT_NUM_SEEDS_PER_ENV = 50
"""The number of seeds to generate for each environment."""


class Task(NamedTuple):
    """
    All data necessary to fully describe a single environment.
    """

    env_name: str
    env_seed: int


class TaskSet(NamedTuple):
    """
    A collection of tasks.
    """

    tasks_dict: dict[str, list[Task]]
    """
    Mapping from environment name to the list of tasks.
    """
    env_names: list[str]
    """
    List of all environment names in the task set.
    """
    env_kwargs_overrides: dict[str, Any]
    """
    Shared kwargs overrides for each environment in the task set.
    Every environment in the task gets the same overrides.
    """


def _generate_task_set(
    env_names: Sequence[str],
    benchmark_seed: int | None,
    num_seeds_per_env: int | None = _DEFAULT_NUM_SEEDS_PER_ENV,
    env_kwargs_overrides: dict[str, dict] | None = None,
) -> TaskSet:
    """Generates seeds for a given set of environments.

    Args:
        env_names: The environment names as a sequence of strings.
        benchmark_seed: The random seed to use for the benchmark.
        num_seeds_per_env: The number of seeds to generate per environment.
        env_kwargs_overrides: The environment kwargs overrides to use for all environments.

    Returns:
        A TaskSet containing all of the generated tasks.
    """

    if env_kwargs_overrides is None:
        env_kwargs_overrides = {}

    env_kwargs_overrides = copy.deepcopy(env_kwargs_overrides)

    tasks_dict: dict[str, list[Task]] = {
        env_name: [] for env_name in env_names
    }

    if num_seeds_per_env == 1 and len(env_names) == 1:
        # Directly use the benchmark seed.
        # This only happens if the user requests a single seed for a single env.
        for env_name in env_names:
            tasks_dict[env_name].append(Task(env_name, benchmark_seed))
    else:
        seed_rng = np.random.default_rng(
            benchmark_seed) if benchmark_seed is not None else np.random.default_rng()

        for env_name in env_names:
            seeds = np.atleast_1d(randint(seed_rng, size=num_seeds_per_env))

            for seed in seeds:
                tasks_dict[env_name].append(Task(env_name, seed))

    return TaskSet(
        tasks_dict=tasks_dict,
        env_names=list(env_names),
        env_kwargs_overrides=env_kwargs_overrides,
    )


class Benchmark:
    def __init__(self,
                 name: str,
                 test_env_names: list[str],
                 train_env_names: list[str],
                 seed: int | None = None,
                 test_train_same_seed: bool = False,
                 env_kwargs_overrides: dict[str, Any] | None = None,
                 num_seeds_per_env: int | None = None):
        self.name = name
        self.test_env_names = test_env_names
        self.train_env_names = train_env_names
        if seed is None:
            self.seed = randint(np.random.default_rng())
        self.seed = seed
        self.test_train_same_seed = test_train_same_seed
        self.env_kwargs_overrides = env_kwargs_overrides
        self.num_seeds_per_env = num_seeds_per_env

    def generate_task_set(self, split: Literal["train", "test"]) -> TaskSet:
        if split == "train":
            actual_seed = self.seed
            env_names = self.train_env_names
        elif split == "test":
            actual_seed = self.seed
            if not self.test_train_same_seed:
                actual_seed += 1
            env_names = self.test_env_names
        else:
            raise ValueError(f"Invalid split: {split}")

        return _generate_task_set(
            env_names,
            actual_seed,
            self.num_seeds_per_env,
            self.env_kwargs_overrides,
        )

    def generate_train_task_set(self) -> TaskSet:
        return self.generate_task_set(split="train")

    def generate_test_task_set(self) -> TaskSet:
        return self.generate_task_set(split="test")


def get_mt1_benchmark(env_name: str, seed: int | None = None, num_seeds_per_env: int | None = None) -> Benchmark:
    """Returns the MT1 benchmark for a given environment name.

    MT1 is a goal-conditioned RL environment for a single Metaworld task.
    A fixed set of seeds is generated.
    The training and testing environments are exactly the same.
    The only purpose of MT1 is to evaluate skill acquisition on a single task.

    Args:
        env_name: The name of the environment.
        seed: The random seed to use for the benchmark.
    Returns:
        The MT1 Benchmark.
    """

    env_names = [env_name]
    return Benchmark(
        name="MT1",
        train_env_names=env_names,
        test_env_names=env_names,
        seed=seed,
        test_train_same_seed=True,
        env_kwargs_overrides=_MT_ENV_KWARGS_OVERRIDE,
        num_seeds_per_env=num_seeds_per_env,
    )


def get_mtX_benchmark(mt_bench: Literal["MT10", "MT25", "MT50"],
                      seed: int | None = None,
                      num_seeds_per_env: int | None = None) -> Benchmark:
    """Returns the MTX benchmark for a given MT benchmark name.

    The MTX benchmarks are multi-task RL environments for multiple Metaworld tasks.
    A fixed set of seeds is generated.
    The training and testing environments are exactly the same.
    The only purpose of MTX is to evaluate skill acquisition across multiple tasks.

    Take a look in `env_dict.py` for the list of tasks in each benchmark.

    Args:
        mt_bench: The name of the MT benchmark.
        seed: The random seed to use for the benchmark.
    Returns:
        The MTX Benchmark.
    """

    if mt_bench not in MT_BENCHMARKS_TRAIN_ENV_NAMES:
        if mt_bench == "MT1":
            raise ValueError(
                "Use `get_mt1_benchmark` to get the MT1 benchmark.")
        raise ValueError(f"Invalid MT benchmark name: {mt_bench}")

    env_names = MT_BENCHMARKS_TRAIN_ENV_NAMES[mt_bench]
    return Benchmark(
        name=mt_bench,
        train_env_names=env_names,
        test_env_names=env_names,
        seed=seed,
        test_train_same_seed=True,
        env_kwargs_overrides=_MT_ENV_KWARGS_OVERRIDE,
        num_seeds_per_env=num_seeds_per_env,
    )


def get_mtCustom_benchmark(
    env_names: list[str],
    seed: int | None = None,
    num_seeds_per_env: int | None = None,
) -> Benchmark:
    """Returns a custom MT benchmark for a given list of environment names.

    The custom MT benchmark is a multi-task RL environment for multiple Metaworld tasks.
    A fixed set of seeds is generated.
    The training and testing environments are exactly the same.
    The only purpose of the custom MT benchmark is to evaluate skill acquisition across multiple tasks.

    Args:
        env_names: The list of environment names.
        seed: The random seed to use for the benchmark.
    Returns:
        The custom MT Benchmark.
    """

    return Benchmark(
        name="custom-mt-envs",
        train_env_names=env_names,
        test_env_names=env_names,
        seed=seed,
        test_train_same_seed=True,
        env_kwargs_overrides=_MT_ENV_KWARGS_OVERRIDE,
        num_seeds_per_env=num_seeds_per_env,
    )


def get_ml1_benchmark(env_name: str, seed: int | None = None, num_seeds_per_env: int | None = None) -> Benchmark:
    """Returns the ML1 benchmark for a given environment name.

    The ML1 benchmark is a goal-conditioned RL environment for a single Metaworld task.
    The train and test environments contain different sets of seeds.
    The only purpose of ML1 is to evaluate meta-learning on a single task.
    The goal position is zeroed out in the observation.

    Args:
        env_name: The name of the environment.
        seed: The random seed to use for the benchmark.
    Returns:
        The ML1 Benchmark.
    """

    env_names = [env_name]
    return Benchmark(
        name="ML1",
        train_env_names=env_names,
        test_env_names=env_names,
        seed=seed,
        test_train_same_seed=False,
        env_kwargs_overrides=_ML_ENV_KWARGS_OVERRIDE,
        num_seeds_per_env=num_seeds_per_env,
    )


def get_mlX_benchmark(ml_bench: Literal["ML10", "ML25", "ML45"],
                      seed: int | None = None,
                      num_seeds_per_env: int | None = None) -> Benchmark:
    """Returns the MLX benchmark for a given ML benchmark name.

    The MLX benchmarks are multi-task RL environments for multiple Metaworld tasks.
    The train and test environments contain different sets of seeds and different tasks.
    The goal position is zeroed out in the observation.

    Take a look in `env_dict.py` for the list of tasks in each benchmark.

    Args:
        ml_bench: The name of the ML benchmark.
        seed: The random seed to use for the benchmark.
    Returns:
        The MLX Benchmark.
    """

    if ml_bench not in ML_BENCHMARKS:
        if ml_bench == "ML1":
            raise ValueError(
                "Use `get_ml1_benchmark` to get the ML1 benchmark.")
        raise ValueError(f"Invalid ML benchmark name: {ml_bench}")

    ml_envs = ML_BENCHMARKS[ml_bench]
    return Benchmark(
        name=ml_bench,
        train_env_names=ml_envs['train'],
        test_env_names=ml_envs['test'],
        seed=seed,
        test_train_same_seed=False,
        env_kwargs_overrides=_ML_ENV_KWARGS_OVERRIDE,
        num_seeds_per_env=num_seeds_per_env,
    )


def get_mlCustom_benchmark(
    train_envs: list[str],
    test_envs: list[str],
    seed: int | None = None,
    num_seeds_per_env: int | None = None,
) -> Benchmark:
    """Returns a custom ML benchmark for given lists of training and testing environment names.

    The custom ML benchmark is a multi-task RL environment for multiple Metaworld tasks.
    The train and test environments contain different sets of seeds and different tasks.
    The goal position is zeroed out in the observation.

    Args:
        train_envs: The list of training environment names.
        test_envs: The list of testing environment names.
        seed: The random seed to use for the benchmark.
    Returns:
        The custom ML Benchmark.
    """

    return Benchmark(
        name="custom-ml-envs",
        train_env_names=train_envs,
        test_env_names=test_envs,
        seed=seed,
        test_train_same_seed=False,
        env_kwargs_overrides=_ML_ENV_KWARGS_OVERRIDE,
        num_seeds_per_env=num_seeds_per_env,
    )
