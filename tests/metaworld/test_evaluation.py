from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest

import metaworld  # noqa: F401
from metaworld import evaluation
from metaworld.policies import ENV_POLICY_MAP
from metaworld.types import QueryableVectorEnv


class ScriptedPolicyAgent(evaluation.MetaLearningAgent):
    def __init__(
        self,
        envs: gym.Env | gym.vector.VectorEnv,
        num_rollouts: int | None = None,
        max_episode_steps: int | None = None,
    ):
        if not isinstance(envs, gym.vector.VectorEnv):
            env = envs
            envs = gym.vector.SyncVectorEnv(
                [lambda: env], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
            )

        env_task_names = evaluation._get_task_names(envs)
        self.policies = [ENV_POLICY_MAP[task]() for task in env_task_names]
        self.num_rollouts = num_rollouts
        self.max_episode_steps = max_episode_steps
        self.adapt_calls = 0
        self.step_calls = 0
        self.resets = 0

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        self.resets += np.sum(env_mask)

    def init(self) -> None:
        return

    def adapt_action(
        self, observations: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]:
        actions: list[npt.NDArray[np.float32]] = []
        num_envs = len(self.policies)
        for env_idx in range(num_envs):
            actions.append(self.policies[env_idx].get_action(observations[env_idx]))
        stacked_actions = np.stack(actions, axis=0, dtype=np.float64)
        return stacked_actions, {
            "log_probs": np.ones((num_envs,)),
            "means": stacked_actions,
            "stds": np.zeros((num_envs,)),
        }

    def step(self, timestep: evaluation.Timestep) -> None:
        assert "log_probs" in timestep.aux_policy_outputs
        assert "means" in timestep.aux_policy_outputs
        assert "stds" in timestep.aux_policy_outputs
        self.step_calls += 1

    def eval_action(
        self, observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        actions: list[npt.NDArray[np.float32]] = []
        num_envs = len(self.policies)
        for env_idx in range(num_envs):
            actions.append(self.policies[env_idx].get_action(observations[env_idx]))
        stacked_actions = np.stack(actions, axis=0, dtype=np.float64)
        return stacked_actions

    def adapt(self) -> None:
        self.adapt_calls += 1


class RemovePartialObservabilityWrapper(QueryableVectorEnv, gym.vector.VectorWrapper):
    env: QueryableVectorEnv

    def get_attr(self, name):
        return self.env.get_attr(name)

    def set_attr(self, name, values):
        return self.env.set_attr(name, values)

    def call(self, name, *args, **kwargs):
        return self.env.call(name, *args, **kwargs)

    def step(self, actions):
        self.env.set_attr("_partially_observable", False)
        return super().step(actions)


def test_evaluation():
    SEED = 1
    max_episode_steps = 300  # To speed up the test
    num_episodes = 50

    random.seed(SEED)
    np.random.seed(SEED)
    envs = gym.make_vec(
        "Meta-World/MT50",
        seed=SEED,
        max_episode_steps=max_episode_steps,
        vector_strategy="async",
    )
    agent = ScriptedPolicyAgent(envs)
    mean_success_rate, mean_returns, success_rate_per_task, _ = evaluation.evaluation(
        agent,
        envs,
        num_episodes=num_episodes,
    )
    assert isinstance(mean_returns, float)
    assert mean_success_rate >= 0.80
    assert len(success_rate_per_task) == envs.num_envs
    failed_tasks: list[tuple[str, float]] = []
    for task_name, success_rate in success_rate_per_task.items():
        if success_rate < 0.80:
            failed_tasks.append((task_name, success_rate))
    if len(failed_tasks) > 0:
        raise AssertionError(f"Not all tasks got > .8 success rate: {failed_tasks}")


@pytest.mark.parametrize("benchmark", ("reach-v3",))
def test_evaluation_mt1(benchmark):
    SEED = 1
    num_episodes = 5

    random.seed(SEED)
    np.random.seed(SEED)
    env = gym.make(
        "Meta-World/MT1",
        env_name=benchmark,
        seed=SEED,
    )
    agent = ScriptedPolicyAgent(env)
    mean_success_rate, mean_returns, success_rate_per_task, _ = evaluation.evaluation(
        agent,
        env,
        num_episodes=num_episodes,
    )
    assert isinstance(mean_returns, float)
    assert set(success_rate_per_task) == {benchmark}
    assert mean_success_rate >= 0.80
    assert success_rate_per_task[benchmark] >= 0.80
    assert np.isclose(mean_success_rate, success_rate_per_task[benchmark])


@pytest.mark.parametrize("benchmark", ("reach-v3",))
def test_evaluation_mt1_vectorized(benchmark):
    SEED = 1
    max_episode_steps = 300
    num_envs = 10
    num_goals = 50
    num_episodes = num_goals // num_envs

    random.seed(SEED)
    np.random.seed(SEED)
    envs = gym.make_vec(
        "Meta-World/MT1-vectorized",
        env_name=benchmark,
        seed=SEED,
        max_episode_steps=max_episode_steps,
        vector_strategy="async",
        num_envs=num_envs,
        num_goals=num_goals,
    )
    assert isinstance(envs, gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv)
    agent = ScriptedPolicyAgent(envs)
    mean_success_rate, mean_returns, success_rate_per_task, _ = evaluation.evaluation(
        agent,
        envs,
        num_episodes=num_episodes,
    )
    assert isinstance(mean_returns, float)
    assert set(success_rate_per_task) == {benchmark}
    assert mean_success_rate >= 0.80
    assert success_rate_per_task[benchmark] >= 0.80
    assert np.isclose(mean_success_rate, success_rate_per_task[benchmark])


# @pytest.mark.skip
@pytest.mark.parametrize("benchmark", ("ML10", "ML45"))
def test_metalearning_evaluation(benchmark):
    SEED = 42

    max_episode_steps = 300
    adaptation_steps = 2  # Number of adaptation iterations
    adaptation_episodes = 2  # Number of train episodes per task in meta_batch_size per adaptation iteration
    num_episodes = 3  # Number of test episodes per task vector

    meta_batch_size = 10  # Number of parallel envs
    goals_per_task = 50  # goal positions for each distinct env class  / task

    random.seed(SEED)
    np.random.seed(SEED)
    envs = gym.make_vec(
        f"Meta-World/{benchmark}-test",
        seed=SEED,
        vector_strategy="async",
        meta_batch_size=meta_batch_size,
        max_episode_steps=max_episode_steps,
        total_tasks_per_cls=goals_per_task,
    )
    envs = RemovePartialObservabilityWrapper(envs)
    agent = ScriptedPolicyAgent(envs, adaptation_episodes, max_episode_steps)

    num_classes = 5  # task size of ML10/45 test set
    num_evals = num_classes * goals_per_task // meta_batch_size

    (
        mean_success_rate,
        mean_returns,
        success_rate_per_task,
    ) = evaluation.metalearning_evaluation(
        agent,
        envs,
        evaluation_episodes=num_episodes,
        adaptation_episodes=adaptation_episodes,
        adaptation_steps=adaptation_steps,
        num_evals=num_evals,
    )
    assert isinstance(mean_returns, float)
    assert mean_success_rate >= 0.80
    assert len(success_rate_per_task) == len(set(evaluation._get_task_names(envs)))
    assert np.all(np.array(list(success_rate_per_task.values())) >= 0.80)
    assert agent.adapt_calls == num_evals * adaptation_steps
    assert (
        agent.step_calls
        == num_evals * adaptation_steps * adaptation_episodes * max_episode_steps
    )


@pytest.mark.parametrize("benchmark", ("reach-v3",))
def test_metalearning_evaluation_ml1(benchmark):
    SEED = 42

    max_episode_steps = 300
    adaptation_steps = 2  # Number of adaptation iterations
    adaptation_episodes = 2  # Number of train episodes per task in meta_batch_size per adaptation iteration
    num_episodes = 3  # Number of test episodes per task vector

    meta_batch_size = 10  # Number of parallel envs
    goals_per_task = 50  # goal positions for each distinct env class  / task

    random.seed(SEED)
    np.random.seed(SEED)
    envs = gym.make_vec(
        "Meta-World/ML1-test",
        env_name=benchmark,
        seed=SEED,
        vector_strategy="async",
        meta_batch_size=meta_batch_size,
        max_episode_steps=max_episode_steps,
        total_tasks_per_cls=goals_per_task,
    )
    envs = RemovePartialObservabilityWrapper(envs)
    agent = ScriptedPolicyAgent(envs, adaptation_episodes, max_episode_steps)

    num_classes = 1  # task size of ML1 test set
    num_evals = num_classes * goals_per_task // meta_batch_size

    (
        mean_success_rate,
        mean_returns,
        success_rate_per_task,
    ) = evaluation.metalearning_evaluation(
        agent,
        envs,
        evaluation_episodes=num_episodes,
        adaptation_episodes=adaptation_episodes,
        adaptation_steps=adaptation_steps,
        num_evals=num_evals,
    )
    assert isinstance(mean_returns, float)
    assert set(success_rate_per_task) == {benchmark}
    assert np.isclose(mean_success_rate, success_rate_per_task[benchmark])
