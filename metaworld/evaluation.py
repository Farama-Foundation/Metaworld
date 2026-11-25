from __future__ import annotations

from typing import NamedTuple, Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from metaworld.env_dict import ALL_V3_ENVIRONMENTS
from metaworld.types import QueryableVectorEnv


class Agent(Protocol):
    def eval_action(
        self, observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None: ...


class MetaLearningAgent(Agent, Protocol):
    def init(self) -> None: ...

    def adapt_action(
        self, observations: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]: ...

    def step(self, timestep: Timestep) -> None: ...

    def adapt(self) -> None: ...


def _get_task_names(
    envs: gym.vector.VectorEnv,
) -> list[str]:
    assert isinstance(envs, QueryableVectorEnv)
    metaworld_cls_to_task_name = {v.__name__: k for k, v in ALL_V3_ENVIRONMENTS.items()}
    return [
        metaworld_cls_to_task_name[task_name]
        for task_name in envs.get_attr("task_name")
    ]


def evaluation(
    agent: Agent,
    eval_envs: gym.Env | gym.vector.VectorEnv,
    num_episodes: int = 50,
) -> tuple[float, float, dict[str, float], dict[str, list[float]]]:
    if not isinstance(eval_envs, gym.vector.VectorEnv):
        eval_env = eval_envs
        eval_envs = gym.vector.SyncVectorEnv(
            [lambda: eval_env], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
        )

    assert isinstance(eval_envs, QueryableVectorEnv)

    terminate_on_success = np.all(eval_envs.get_attr("terminate_on_success")).item()
    eval_envs.call("toggle_terminate_on_success", True)

    obs: npt.NDArray[np.float64]
    obs, _ = eval_envs.reset()
    agent.reset(np.ones(eval_envs.num_envs, dtype=np.bool_))

    env_successes = np.zeros(eval_envs.num_envs)
    env_episodic_returns: dict[int, list[float]] = {
        i: [] for i in range(eval_envs.num_envs)
    }

    def eval_done(returns):
        return all(len(r) >= num_episodes for _, r in returns.items())

    while not eval_done(env_episodic_returns):
        actions = agent.eval_action(obs)
        obs, _, terminations, truncations, infos = eval_envs.step(actions)

        dones = np.logical_or(terminations, truncations)
        agent.reset(dones)

        for i, env_ended in enumerate(dones):
            if env_ended:
                final_info = infos.get("final_info", infos)
                episode_return = float(final_info["episode"]["r"][i])
                success = int(final_info["success"][i])

                env_episodic_returns[i].append(episode_return)

                if len(env_episodic_returns[i]) <= num_episodes:
                    env_successes[i] += success

    # Main statistics are over the batch of eval envs only
    episodic_returns = {
        i: returns[:num_episodes] for i, returns in env_episodic_returns.items()
    }
    success_rate_per_env = env_successes / num_episodes

    mean_success_rate = float(np.mean(success_rate_per_env))
    mean_returns = float(np.mean(list(episodic_returns.values())))

    # Env class / task statistics for logging
    task_names = _get_task_names(eval_envs)
    task_names_unique = list(dict.fromkeys(task_names))
    task_env_indices: dict[str, list[int]] = {
        task_name: [] for task_name in task_names_unique
    }
    for env_idx, task_name in enumerate(task_names):
        task_env_indices[task_name].append(env_idx)
    success_rate_per_task = {}
    episodic_returns_per_task = {}
    for task_name in task_names_unique:
        env_indices = task_env_indices[task_name]
        success_rate_per_task[task_name] = float(
            np.sum(success_rate_per_env[env_indices]) / len(env_indices)
        )
        episodic_returns_per_task[task_name] = [
            episode_return
            for env_idx in env_indices
            for episode_return in episodic_returns[env_idx]
        ]

    eval_envs.call("toggle_terminate_on_success", terminate_on_success)

    return (
        mean_success_rate,
        mean_returns,
        success_rate_per_task,
        episodic_returns_per_task,
    )


def metalearning_evaluation(
    agent: MetaLearningAgent,
    eval_envs: gym.vector.VectorEnv,
    num_evals: int = 10,  # Assuming 40 goals per test task and meta batch size of 20
    adaptation_steps: int = 1,
    adaptation_episodes: int = 10,
    evaluation_episodes: int = 3,
) -> tuple[float, float, dict[str, float]]:
    assert isinstance(eval_envs, QueryableVectorEnv)

    eval_envs.call("toggle_sample_tasks_on_reset", False)
    eval_envs.call("toggle_terminate_on_success", False)
    task_names_unique = list(dict.fromkeys(_get_task_names(eval_envs)))

    total_mean_success_rate = 0.0
    total_mean_return = 0.0
    success_rate_per_task = np.zeros((num_evals, len(task_names_unique)))

    for i in range(num_evals):
        obs: npt.NDArray[np.float64]

        eval_envs.call("sample_tasks")
        agent.init()

        for _ in range(adaptation_steps):
            obs, _ = eval_envs.reset()
            episodes_elapsed = np.zeros((eval_envs.num_envs,), dtype=np.uint16)

            while not (episodes_elapsed >= adaptation_episodes).all():
                actions, aux_policy_outs = agent.adapt_action(obs)
                next_obs, rewards, terminations, truncations, _ = eval_envs.step(
                    actions
                )
                agent.step(
                    Timestep(
                        obs,
                        actions,
                        rewards,
                        terminations,
                        truncations,
                        aux_policy_outs,
                    )
                )
                episodes_elapsed += np.logical_or(terminations, truncations)
                obs = next_obs

            agent.adapt()

        # Evaluation
        mean_success_rate, mean_return, _success_rate_per_task, _ = evaluation(
            agent, eval_envs, evaluation_episodes
        )
        total_mean_success_rate += mean_success_rate
        total_mean_return += mean_return
        success_rate_per_task[i] = np.array(
            [_success_rate_per_task[task_name] for task_name in task_names_unique]
        )

    # Env class / task success rates for logs only
    mean_success_rate_per_task = (success_rate_per_task).mean(axis=0)
    task_success_rates = {
        task_name: mean_success_rate_per_task[i]
        for i, task_name in enumerate(task_names_unique)
    }

    return (
        total_mean_success_rate / num_evals,
        total_mean_return / num_evals,
        task_success_rates,
    )


class Timestep(NamedTuple):
    observation: npt.NDArray
    action: npt.NDArray
    reward: npt.NDArray
    terminated: npt.NDArray
    truncated: npt.NDArray
    aux_policy_outputs: dict[str, npt.NDArray]
