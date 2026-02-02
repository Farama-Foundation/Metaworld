from __future__ import annotations

from typing import NamedTuple, Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from metaworld.env_dict import ALL_V3_ENVIRONMENTS, envs_get_env_names


class Agent(Protocol):
    def eval_action(
        self, observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        ...

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        ...


class MetaLearningAgent(Agent, Protocol):
    def init(self) -> None:
        ...

    def adapt_action(
        self, observations: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]:
        ...

    def step(self, timestep: Timestep) -> None:
        ...

    def adapt(self) -> None:
        ...


def evaluation(
    agent: Agent,
    eval_envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    num_episodes: int = 50,
) -> tuple[float, float, dict[str, float], dict[str, list[float]]]:
    terminate_on_success = np.all(
        eval_envs.get_attr("terminate_on_success")).item()
    eval_envs.call("toggle_terminate_on_success", True)

    obs: npt.NDArray[np.float64]
    obs, _ = eval_envs.reset()
    agent.reset(np.ones(eval_envs.num_envs, dtype=np.bool_))

    env_names = envs_get_env_names(eval_envs)
    successes = {env_name: 0 for env_name in set(env_names)}
    episodic_returns: dict[str, list[float]] = {
        env_name: [] for env_name in set(env_names)
    }

    pbar_envs_done = tqdm(total=len(set(task_names)) *
                          num_episodes, desc="Evaluation")

    def eval_done(returns):
        return all(len(r) >= num_episodes for _, r in returns.items())

    while not eval_done(episodic_returns):
        actions = agent.eval_action(obs)
        obs, _, terminations, truncations, infos = eval_envs.step(actions)

        dones = np.logical_or(terminations, truncations)
        agent.reset(dones)

        for i, env_ended in enumerate(dones):
            if env_ended:
                env = env_names[i]
                current_count = len(episodic_returns[env])
                if current_count < num_episodes:
                    pbar_envs_done.update(1)
                episodic_returns[env].append(
                    float(infos["final_info"]["episode"]["r"][i])
                )
                if len(episodic_returns[env]) <= num_episodes:
                    successes[env] += int(infos["final_info"]["success"][i])

    episodic_returns = {
        env_name: returns[:num_episodes]
        for env_name, returns in episodic_returns.items()
    }

    success_rate_per_task = {
        env_name: task_successes / num_episodes
        for env_name, task_successes in successes.items()
    }
    mean_success_rate = np.mean(list(success_rate_per_task.values()))
    mean_returns = np.mean(list(episodic_returns.values()))

    eval_envs.call("toggle_terminate_on_success", terminate_on_success)

    return (
        float(mean_success_rate),
        float(mean_returns),
        success_rate_per_task,
        episodic_returns,
    )


def metalearning_evaluation(
    agent: MetaLearningAgent,
    eval_envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    num_evals: int = 10,  # Assuming 40 goals per test task and meta batch size of 20
    adaptation_steps: int = 1,
    adaptation_episodes: int = 10,
    evaluation_episodes: int = 3,
) -> tuple[float, float, dict[str, float]]:
    eval_envs.call("toggle_sample_tasks_on_reset", False)
    eval_envs.call("toggle_terminate_on_success", False)
    env_names = envs_get_env_names(eval_envs)

    total_mean_success_rate = 0.0
    total_mean_return = 0.0
    success_rate_per_task = np.zeros((num_evals, len(set(env_names))))

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
            list(_success_rate_per_task.values()))

    success_rates = (success_rate_per_task).mean(axis=0)
    env_success_rates = {
        env_name: success_rates[i] for i, env_name in enumerate(set(env_names))
    }

    return (
        total_mean_success_rate / num_evals,
        total_mean_return / num_evals,
        env_success_rates,
    )


class Timestep(NamedTuple):
    observation: npt.NDArray
    action: npt.NDArray
    reward: npt.NDArray
    terminated: npt.NDArray
    truncated: npt.NDArray
    aux_policy_outputs: dict[str, npt.NDArray]
