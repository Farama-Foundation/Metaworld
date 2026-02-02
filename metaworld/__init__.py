"""The public-facing Metaworld API."""

from __future__ import annotations

from functools import partial
from typing import Any, Literal, Sequence

import gymnasium as gym  # type: ignore
import numpy as np

# noqa: D104
from gymnasium.envs.registration import register

from metaworld.env_dict import (
    ALL_V3_ENVIRONMENTS,
    MT_BENCHMARKS_TRAIN_ENV_NAMES,
    ML_BENCHMARKS,
)
from metaworld.sawyer_xyz_env import SawyerXYZEnv  # type: ignore
from metaworld.types import Task, TaskSet  # type: ignore
from metaworld.utils.numpy import randint
from metaworld.wrappers import (
    AutoTerminateOnSuccessWrapper,
    CheckpointWrapper,
    NormalizeRewardsExponential,
    OneHotWrapper,
    PseudoRandomTaskSelectWrapper,
    RandomTaskSelectWrapper,
    RNNBasedMetaRLWrapper,
)

_ML_ENV_KWARGS_OVERRIDE = dict(goal_observable=False)
"""The overrides for the Meta-Learning benchmarks. Disables the inclusion of the goal position in the observation."""

_MT_ENV_KWARGS_OVERRIDE = dict(goal_observable=True)
"""The overrides for the Multi-Task benchmarks. Enables the inclusion of the goal position in the observation."""

_DEFAULT_NUM_SEEDS_PER_ENV = 50
"""The number of seeds to generate for each environment."""


def _generate_task_set(
    env_names: Sequence[str],
    benchmark_seed: int | None,
    num_seeds_per_env: int | None = _DEFAULT_NUM_SEEDS_PER_ENV,
) -> TaskSet:
    """Generates seeds for a given set of environments.

    Args:
        env_names: The environment names as a sequence of strings.
        benchmark_seed: The random seed to use for the benchmark.
        num_seeds_per_env: The number of seeds to generate per environment.

    Returns:
        A TaskSet containing all of the generated tasks.
    """

    seed_rng = np.random.default_rng(
        benchmark_seed) if benchmark_seed is not None else np.random.default_rng()

    tasks_dict: dict[str, list[Task]] = {
        env_name: [] for env_name in env_names
    }

    for env_name in env_names:
        seeds = np.atleast_1d(randint(seed_rng, size=num_seeds_per_env))

        for seed in seeds:
            tasks_dict[env_name].append(Task(env_name, seed))

    return TaskSet(
        tasks_dict=tasks_dict,
        env_names=list(env_names),
    )


def _init_env_with_wrappers(
    env_name: str,
    tasks: list[Task],
    max_episode_steps: int = 500,
    terminate_on_success: bool = True,
    use_one_hot: bool = False,
    env_id: int | None = None,
    num_env_ids: int | None = None,
    recurrent_info_in_obs: bool = False,
    normalize_reward_in_recurrent_info: bool = True,
    task_select: Literal["random", "pseudorandom"] = "random",
    reward_normalization_method: Literal["gymnasium",
                                         "exponential"] | None = None,
    normalize_observations: bool = False,
    reward_alpha: float = 0.001,
    **extra_env_kwargs: dict[str, Any],
) -> gym.Env:
    env_cls = ALL_V3_ENVIRONMENTS[env_name]

    # All tasks have the same env kwargs, so we can just use the first one
    kw_args = extra_env_kwargs.copy()

    env: gym.Env = env_cls(
        max_episode_steps=max_episode_steps,
        **kw_args,
    )
    env = gym.wrappers.TimeLimit(
        env, max_episode_steps)  # type: ignore
    env = AutoTerminateOnSuccessWrapper(env)
    env.toggle_terminate_on_success(terminate_on_success)
    if use_one_hot:
        if env_id is None or num_env_ids is None:
            raise ValueError(
                "env_id and num_env_ids must be provided when using one-hot encoding."
            )
        env = OneHotWrapper(env, env_id, num_env_ids)
    if recurrent_info_in_obs:
        env = RNNBasedMetaRLWrapper(
            env, normalize_reward=normalize_reward_in_recurrent_info
        )
    if reward_normalization_method == "gymnasium":
        env = gym.wrappers.NormalizeReward(env)
    elif reward_normalization_method == "exponential":
        env = NormalizeRewardsExponential(reward_alpha=reward_alpha, env=env)
    if normalize_observations:
        env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if task_select != "random":
        env = PseudoRandomTaskSelectWrapper(env, tasks)
    else:
        env = RandomTaskSelectWrapper(env, tasks)

    env = CheckpointWrapper(env, f"{env_cls}_{env_id}")
    return env


def _vectorizer_from_strategy(
    vector_strategy: Literal["sync",
                             "async"] | type[gym.vector.VectorEnv] = "sync",
) -> type[gym.vector.VectorEnv]:
    vectorizer: type[gym.vector.VectorEnv]
    if vector_strategy == "sync":
        vectorizer = gym.vector.SyncVectorEnv
    elif vector_strategy == "async":
        vectorizer = gym.vector.AsyncVectorEnv
    else:
        vectorizer = vector_strategy
    return vectorizer


def _vectorize_task_set(
    task_set: TaskSet,
    meta_batch_size: int,
    vector_strategy: Literal["sync",
                             "async"] | type[gym.vector.VectorEnv] | None = None,
    autoreset_mode: gym.vector.AutoresetMode | str = gym.vector.AutoresetMode.SAME_STEP,
    **kwargs,
) -> gym.vector.VectorEnv:
    assert (
        meta_batch_size % len(task_set.env_names) == 0
    ), "meta_batch_size must be divisible by the environment count"
    tasks_per_env = meta_batch_size // len(task_set.env_names)

    tasks_per_parallel_env = []
    for env_name in task_set.env_names:
        # Filter tasks for this env name
        tasks = task_set.tasks_dict[env_name]
        # Split tasks into `tasks_per_env` sublists
        subenv_tasks = [tasks[i::tasks_per_env]
                        for i in range(0, tasks_per_env)]
        for tasks_for_subenv in subenv_tasks:
            assert (
                len(tasks_for_subenv) == len(tasks) // tasks_per_env
            ), f"Invalid division of subtasks, expected {len(tasks) // tasks_per_env} got {len(tasks_for_subenv)}"
            tasks_per_parallel_env.append((env_name, tasks_for_subenv))

    env_name_to_id = {name: i for i, name in enumerate(task_set.env_names)}

    vectorizer = _vectorizer_from_strategy(vector_strategy)
    return vectorizer(
        [
            partial(
                _init_env_with_wrappers,
                env_name=env_name,
                env_id=env_name_to_id[env_name],
                num_env_ids=len(task_set.env_names),
                tasks=tasks,
                **kwargs,
            )
            for env_name, tasks in tasks_per_parallel_env
        ],
        autoreset_mode=autoreset_mode,
    )


def _mt1_entry_point(
    env_name: str,
    seed: int | None = None,
    num_seeds_per_env: int | None = None,
    **kwargs,
):
    if num_seeds_per_env == 1:
        # Patch the seed directly into the environment
        tasks = [Task(env_name, seed)]
    else:
        tasks = list(_generate_task_set(
            [env_name], seed, num_seeds_per_env
        ).tasks_dict[env_name])
    return _init_env_with_wrappers(
        env_name=env_name,
        tasks=tasks,
        **kwargs,
    )


def _mt1_vector_entry_point(
    env_name: str,
):
    return _mtX_vector_entry_point(
        env_names=[env_name],
    )


def _mtX_vector_entry_point(
    env_names: list[str],
    seed: int | None = None,
    num_seeds_per_env: int | None = None,
    meta_batch_size: int | None = None,
    **kwargs,
) -> gym.Env | gym.vector.VectorEnv:
    task_set = _generate_task_set(
        env_names,
        seed,
        num_seeds_per_env,
    )

    num_env_ids = len(task_set.env_names)

    if meta_batch_size is None:
        meta_batch_size = num_env_ids

    return _vectorize_task_set(
        task_set,
        meta_batch_size=meta_batch_size,
        **kwargs,
    )


def _ml1_vector_entry_point(
    env_name: str,
    seed: int | None = None,
    split: Literal["train", "test"] = "train",
    **kwargs,
):
    if seed is not None and split == "test":
        seed = seed + 1

    return _mtX_vector_entry_point(
        env_names=[env_name],
        seed=seed,
        **kwargs,
    )


def _custom_ml_vector_entry_point(
    train_envs: list[str],
    test_envs: list[str],
    split: Literal["train", "test"] = "train",
    **kwargs,
):
    env_names = train_envs if split == "train" else test_envs
    return _mtX_vector_entry_point(
        env_names=env_names,
        **kwargs,
    )


def _register_mw_envs() -> None:

    # --- MT Envs ---

    register(
        id="Meta-World/MT1",
        entry_point=_mt1_entry_point,
        vector_entry_point=_mt1_vector_entry_point,
        kwargs=_MT_ENV_KWARGS_OVERRIDE,
    )

    for mt_bench in ["MT10", "MT25", "MT50"]:
        register(
            id=f"Meta-World/{mt_bench}",
            vector_entry_point=partial(
                _mtX_vector_entry_point,
                env_names=MT_BENCHMARKS_TRAIN_ENV_NAMES[mt_bench],
            ),
            kwargs=_MT_ENV_KWARGS_OVERRIDE,
        )

    register(
        id="Meta-World/custom-mt-envs",
        vector_entry_point=_mtX_vector_entry_point,
        kwargs=_MT_ENV_KWARGS_OVERRIDE,
    )

    # --- ML Envs ---

    for split in ["train", "test"]:
        register(
            id=f"Meta-World/ML1-{split}",
            vector_entry_point=partial(_ml1_vector_entry_point, split),
            kwargs=_ML_ENV_KWARGS_OVERRIDE,
        )

    for ml_bench in ["ML10", "ML25", "ML45"]:
        for split in ["train", "test"]:
            register(
                id=f"Meta-World/{ml_bench}-{split}",
                vector_entry_point=partial(
                    _mtX_vector_entry_point,
                    env_names=ML_BENCHMARKS[ml_bench][split],
                ),
                kwargs=_ML_ENV_KWARGS_OVERRIDE,
            )

    register(
        id="Meta-World/custom-ml-envs",
        vector_entry_point=_custom_ml_vector_entry_point,
        kwargs=_ML_ENV_KWARGS_OVERRIDE,
    )


_register_mw_envs()
__all__: list[str] = []
