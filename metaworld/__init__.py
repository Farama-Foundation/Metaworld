"""The public-facing Metaworld API."""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import gymnasium as gym  # type: ignore

# noqa: D104
from gymnasium.envs.registration import register

from metaworld.benchmark import (
    get_mt1_benchmark,
    get_mtX_benchmark,
    get_mlCustom_benchmark,
    get_ml1_benchmark,
    get_mlX_benchmark,
    get_mtCustom_benchmark,
    TaskSet,
    Task,
)
from metaworld.env_dict import (
    ALL_V3_ENVIRONMENTS,
)
from metaworld.sawyer_xyz_env import SawyerXYZEnv  # type: ignore
from metaworld.wrappers import (
    AutoTerminateOnSuccessWrapper,
    CheckpointWrapper,
    NormalizeRewardsExponential,
    OneHotWrapper,
    PseudoRandomTaskSelectWrapper,
    RandomTaskSelectWrapper,
    RNNBasedMetaRLWrapper,
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
    meta_batch_size: int | None = None,
    vector_strategy: Literal["sync",
                             "async"] | type[gym.vector.VectorEnv] | None = None,
    autoreset_mode: gym.vector.AutoresetMode | str = gym.vector.AutoresetMode.SAME_STEP,
    **kwargs,
) -> gym.vector.VectorEnv:

    num_env_ids = len(task_set.env_names)

    if meta_batch_size is None:
        meta_batch_size = num_env_ids

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
                **task_set.env_kwargs_overrides,
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
    mt1_benchmark = get_mt1_benchmark(
        env_name,
        seed,
        num_seeds_per_env,
    )
    task_set = mt1_benchmark.generate_train_task_set()
    tasks = task_set.tasks_dict[env_name]
    return _init_env_with_wrappers(
        env_name=env_name,
        tasks=tasks,
        **task_set.env_kwargs_overrides,
        **kwargs,
    )


def _mtX_vector_entry_point(
    mt_bench: Literal["MT10", "MT25", "MT50"],
    seed: int | None = None,
    num_seeds_per_env: int | None = None,
    **kwargs,
) -> gym.Env | gym.vector.VectorEnv:
    mtX_benchmark = get_mtX_benchmark(
        mt_bench,
        seed,
        num_seeds_per_env,
    )

    task_set = mtX_benchmark.generate_train_task_set()

    return _vectorize_task_set(
        task_set,
        **kwargs,
    )


def _mtCustom_vector_entry_point(
    train_envs: list[str],
    seed: int | None = None,
    num_seeds_per_env: int | None = None,
    **kwargs,
) -> gym.Env | gym.vector.VectorEnv:
    mtX_benchmark = get_mtCustom_benchmark(
        train_envs,
        seed,
        num_seeds_per_env,
    )

    task_set = mtX_benchmark.generate_train_task_set()

    return _vectorize_task_set(
        task_set,
        **kwargs,
    )


def _ml1_vector_entry_point(
    env_name: str,
    seed: int | None = None,
    split: Literal["train", "test"] = "train",
    num_seeds_per_env: int | None = None,
    **kwargs,
):
    ml1_benchmark = get_ml1_benchmark(
        env_name,
        seed,
        num_seeds_per_env,
    )

    task_set = ml1_benchmark.generate_task_set(split=split)

    return _vectorize_task_set(
        task_set,
        **kwargs,
    )


def _mlX_vector_entry_point(
    ml_bench: Literal["ML10", "ML25", "ML45"],
    seed: int | None = None,
    split: Literal["train", "test"] = "train",
    num_seeds_per_env: int | None = None,
    **kwargs,
):
    mlX_benchmark = get_mlX_benchmark(
        ml_bench,
        seed,
        num_seeds_per_env,
    )

    task_set = mlX_benchmark.generate_task_set(split=split)

    return _vectorize_task_set(
        task_set,
        **kwargs,
    )


def _mlCustom_vector_entry_point(
    train_envs: list[str],
    test_envs: list[str],
    seed: int | None = None,
    split: Literal["train", "test"] = "train",
    num_seeds_per_env: int | None = None,
    **kwargs,
):
    mlX_benchmark = get_mlCustom_benchmark(
        train_envs,
        test_envs,
        seed,
        num_seeds_per_env,
    )

    task_set = mlX_benchmark.generate_task_set(split=split)

    return _vectorize_task_set(
        task_set,
        **kwargs,
    )


def _register_mw_envs() -> None:

    # --- MT Envs ---

    register(
        id="Meta-World/MT1",
        entry_point=_mt1_entry_point,
    )

    for mt_bench in ["MT10", "MT25", "MT50"]:
        register(
            id=f"Meta-World/{mt_bench}",
            vector_entry_point=partial(
                _mtX_vector_entry_point,
                mt_bench=mt_bench,
            ),
        )

    register(
        id="Meta-World/custom-mt-envs",
        vector_entry_point=_mtCustom_vector_entry_point,
    )

    # --- ML Envs ---

    for split in ["train", "test"]:
        register(
            id=f"Meta-World/ML1-{split}",
            vector_entry_point=partial(
                _ml1_vector_entry_point,
                split=split
            ),
        )

    for ml_bench in ["ML10", "ML25", "ML45"]:
        for split in ["train", "test"]:
            register(
                id=f"Meta-World/{ml_bench}-{split}",
                vector_entry_point=partial(
                    _mlX_vector_entry_point,
                    ml_bench=ml_bench,
                    split=split
                ),
            )

    register(
        id="Meta-World/custom-ml-envs",
        vector_entry_point=_mlCustom_vector_entry_point,
    )


_register_mw_envs()
__all__: list[str] = []
