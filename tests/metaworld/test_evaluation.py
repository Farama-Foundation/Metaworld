from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest

import metaworld  # noqa: F401
from metaworld import evaluation
from metaworld.policies import ENV_POLICY_MAP


class ScriptedPolicyAgent(evaluation.MetaLearningAgent):
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
        num_rollouts: int | None = None,
        max_episode_steps: int | None = None,
    ):
        env_task_names = evaluation._get_task_names(envs)
        self.policies = [ENV_POLICY_MAP[task]() for task in env_task_names]  # type: ignore
        self.num_rollouts = num_rollouts
        self.max_episode_steps = max_episode_steps
        self.adapt_calls = 0
        self.step_calls = 0
        self.resets = 0

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        self.resets += np.sum(env_mask)  # type: ignore

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


class RemovePartialObservabilityWrapper(gym.vector.VectorWrapper):
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
    SEED = 42
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
    num_envs = envs.num_envs
    agent = ScriptedPolicyAgent(envs)
    mean_success_rate, mean_returns, success_rate_per_task, _ = evaluation.evaluation(
        agent, envs, num_episodes=num_episodes
    )
    envs.close()
    assert isinstance(mean_returns, float)
    assert len(success_rate_per_task) == num_envs
    worst_accepted_fail_rate = 0.8
    failed_envs_names = []
    failed_envs_rates = []
    for task_name, success_rate in success_rate_per_task.items():
        if success_rate < worst_accepted_fail_rate:
            failed_envs_names.append(task_name)
            failed_envs_rates.append(success_rate)
    if len(failed_envs_names) > 0:
        print(
            f"The following environments failed the success rate threshold of {worst_accepted_fail_rate*100}%:"
        )
        for name, rate in zip(failed_envs_names, failed_envs_rates):
            print(f"- {name}: {rate*100}% success rate")
        assert False, "Some environments did not meet the success rate threshold."
    assert mean_success_rate >= worst_accepted_fail_rate


# @pytest.mark.skip
@pytest.mark.parametrize("benchmark", ("ML10", "ML45"))
def test_metalearning_evaluation(benchmark):
    SEED = 42

    max_episode_steps = 300
    meta_batch_size = 10  # Number of parallel envs

    adaptation_steps = 2  # Number of adaptation iterations
    adaptation_episodes = 2  # Number of train episodes per task in meta_batch_size per adaptation iteration
    num_evals = 50  # Number of different task vectors tested for each task
    num_episodes = 1  # Number of test episodes per task vector

    random.seed(SEED)
    np.random.seed(SEED)
    envs = gym.make_vec(
        f"Meta-World/{benchmark}-test",
        seed=SEED,
        vector_strategy="async",
        meta_batch_size=meta_batch_size,
        max_episode_steps=max_episode_steps,
    )
    envs = RemovePartialObservabilityWrapper(envs)
    agent = ScriptedPolicyAgent(envs, adaptation_episodes, max_episode_steps)
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
