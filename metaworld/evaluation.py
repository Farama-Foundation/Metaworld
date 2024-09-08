from __future__ import annotations

import time
from typing import Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt


class Agent(Protocol):
    def get_action_eval(obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ...


class MetaLearningAgent(Agent):
    def adapt(rollouts: npt.NDArray[np.float64]) -> None:
        ...


def evaluation(
    agent: Agent,
    eval_envs: gym.vector.VectorEnv,
    num_episodes: int,
    task_names: list[str] | None = None,
) -> tuple[float, float, npt.NDArray]:
    print(f"Evaluating for {num_episodes} episodes.")
    obs, _ = eval_envs.reset()
    if task_names is not None:
        successes = {task_name: 0 for task_name in set(task_names)}
        episodic_returns = {task_name: [] for task_name in set(task_names)}
        envs_per_task = {
            task_name: task_names.count(task_name) for task_name in set(task_names)
        }
    else:
        successes = np.zeros(eval_envs.num_envs)
        episodic_returns = [[] for _ in range(eval_envs.num_envs)]

    start_time = time.time()

    def eval_done(returns):
        if isinstance(returns, dict):
            return all(
                len(r) >= (num_episodes * envs_per_task[task_name])
                for task_name, r in returns.items()
            )
        else:
            return all(len(r) >= num_episodes for r in returns)

    while not eval_done(episodic_returns):
        actions = agent.get_action_eval(obs)
        obs, _, terminations, truncations, infos = eval_envs.step(actions)
        for i, env_ended in enumerate(np.logical_or(terminations, truncations)):
            if env_ended:
                if task_names is not None:
                    episodic_returns[task_names[i]].append(
                        float(infos["episode"]["r"][i])
                    )
                    if (
                        len(episodic_returns[task_names[i]])
                        <= num_episodes * envs_per_task[task_names[i]]
                    ):
                        successes[task_names[i]] += int(infos["success"][i])
                else:
                    episodic_returns[i].append(float(infos["episode"]["r"][i]))
                    if len(episodic_returns[i]) <= num_episodes:
                        successes[i] += int(infos["success"][i])

    if isinstance(episodic_returns, dict):
        episodic_returns = {
            task_name: returns[: (num_episodes * envs_per_task[task_name])]
            for task_name, returns in episodic_returns.items()
        }
    else:
        episodic_returns = [returns[:num_episodes] for returns in episodic_returns]

    print(f"Evaluation time: {time.time() - start_time:.2f}s")

    if isinstance(successes, dict):
        success_rate_per_task = np.array(
            [
                successes[task_name] / (num_episodes * envs_per_task[task_name])
                for task_name in set(task_names)
            ]
        )
        mean_success_rate = np.mean(success_rate_per_task)
        mean_returns = np.mean(list(episodic_returns.values()))
    else:
        success_rate_per_task = successes / num_episodes
        mean_success_rate = np.mean(success_rate_per_task)
        mean_returns = np.mean(episodic_returns)

    return mean_success_rate, mean_returns, success_rate_per_task


def metalearning_evaluation(
    agent: MetaLearningAgent,
    eval_envs: gym.vector.VectorEnv,
    adaptation_steps: int,
    max_episode_steps: int,
    adaptation_episodes: int,
    eval_episodes: int,
    num_evals: int,
    buffer_kwargs: dict,  # TODO
    task_names: list[str] | None = None,
):
    total_mean_success_rate = 0.0
    total_mean_return = 0.0

    if task_names is not None:
        success_rate_per_task = np.zeros((num_evals, len(set(task_names))))
    else:
        success_rate_per_task = np.zeros((num_evals, eval_envs.num_envs))

    for i in range(num_evals):
        eval_envs.call("toggle_sample_tasks_on_reset", False)
        eval_envs.call("toggle_terminate_on_success", False)
        obs, _ = zip(*eval_envs.call("sample_tasks"))
        obs = np.stack(obs)
        # TODO
        eval_buffer = MultiTaskRolloutBuffer(
            num_tasks=eval_envs.num_envs,
            rollouts_per_task=adaptation_episodes,
            max_episode_steps=max_episode_steps,
        )

        for _ in range(adaptation_steps):
            while not eval_buffer.ready:
                action, log_probs, means, stds, key = agent.get_action_eval(obs)
                next_obs, reward, _, truncated, _ = eval_envs.step(action)
                eval_buffer.push(obs, action, reward, truncated, log_probs, means, stds)
                obs = next_obs

            rollouts = eval_buffer.get(**buffer_kwargs)
            agent.adapt(rollouts)
            eval_buffer.reset()

        # Evaluation
        eval_envs.call("toggle_terminate_on_success", True)
        mean_success_rate, mean_return, _success_rate_per_task = evaluation(
            agent, eval_envs, eval_episodes, task_names
        )
        total_mean_success_rate += mean_success_rate
        total_mean_return += mean_return
        success_rate_per_task[i] = _success_rate_per_task

    success_rates = (success_rate_per_task).mean(axis=0)
    task_success_rates = {
        task_name: success_rates[i] for i, task_name in enumerate(set(task_names))
    }

    return (
        total_mean_success_rate / num_evals,
        total_mean_return / num_evals,
        task_success_rates,
        key,
    )
