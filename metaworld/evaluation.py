from __future__ import annotations

from typing import NamedTuple, Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from metaworld.env_dict import ALL_V3_ENVIRONMENTS


class Agent(Protocol):
    def eval_action(
        self, observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        ...


class MetaLearningAgent(Agent, Protocol):
    def reset_state(self) -> None:
        ...

    def adapt_action(  # type: ignore[empty-body]
        self, observations: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]:
        ...

    def adapt(self, rollouts: Rollout) -> None:
        ...


def _get_task_names(
    envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
) -> list[str]:
    metaworld_cls_to_task_name = {v.__name__: k for k, v in ALL_V3_ENVIRONMENTS.items()}
    return [
        metaworld_cls_to_task_name[task_name]
        for task_name in envs.get_attr("task_name")
    ]


def evaluation(
    agent: Agent,
    eval_envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    num_episodes: int = 50,
) -> tuple[float, float, dict[str, float], dict[str, list[float]]]:
    terminate_on_success = np.all(eval_envs.get_attr("terminate_on_success")).item()
    eval_envs.call("toggle_terminate_on_success", True)

    obs: npt.NDArray[np.float64]
    obs, _ = eval_envs.reset()
    task_names = _get_task_names(eval_envs)
    successes = {task_name: 0 for task_name in set(task_names)}
    episodic_returns: dict[str, list[float]] = {
        task_name: [] for task_name in set(task_names)
    }

    def eval_done(returns):
        return all(len(r) >= num_episodes for _, r in returns.items())

    while not eval_done(episodic_returns):
        actions = agent.eval_action(obs)
        obs, _, terminations, truncations, infos = eval_envs.step(actions)
        for i, env_ended in enumerate(np.logical_or(terminations, truncations)):
            if env_ended:
                episodic_returns[task_names[i]].append(float(infos["episode"]["r"][i]))
                if len(episodic_returns[task_names[i]]) <= num_episodes:
                    successes[task_names[i]] += int(infos["success"][i])

    episodic_returns = {
        task_name: returns[:num_episodes]
        for task_name, returns in episodic_returns.items()
    }

    success_rate_per_task = {
        task_name: task_successes / num_episodes
        for task_name, task_successes in successes.items()
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
    max_episode_steps: int = 500,
    evaluation_episodes: int = 3,
) -> tuple[float, float, dict[str, float]]:
    task_names = _get_task_names(eval_envs)

    total_mean_success_rate = 0.0
    total_mean_return = 0.0

    success_rate_per_task = np.zeros((num_evals, len(set(task_names))))

    eval_buffer = _MultiTaskRolloutBuffer(
        num_tasks=eval_envs.num_envs,
        rollouts_per_task=adaptation_episodes,
        max_episode_steps=max_episode_steps,
        envs=eval_envs,
    )

    for i in range(num_evals):
        obs: npt.NDArray[np.float64]

        eval_envs.call("toggle_sample_tasks_on_reset", False)
        eval_envs.call("toggle_terminate_on_success", False)
        eval_envs.call("sample_tasks")
        agent.reset_state()

        for _ in range(adaptation_steps):
            obs, _ = eval_envs.reset()
            eval_buffer.reset()
            has_autoreset = np.full((eval_envs.num_envs,), False)
            while not eval_buffer.ready:
                actions, aux_policy_outs = agent.adapt_action(obs)
                next_obs: npt.NDArray[np.float64]
                rewards: npt.NDArray[np.float64]
                next_obs, rewards, terminations, truncations, _ = eval_envs.step(
                    actions
                )
                if not has_autoreset.any():
                    eval_buffer.add(
                        obs,
                        action=actions,
                        reward=rewards,
                        done=truncations,
                        log_prob=aux_policy_outs.get("log_probs"),
                        mean=aux_policy_outs.get("means"),
                        std=aux_policy_outs.get("stds"),
                        value=aux_policy_outs.get("values"),
                    )
                has_autoreset = np.logical_or(terminations, truncations)
                obs = next_obs

            agent.adapt(eval_buffer.get())

        # Evaluation
        mean_success_rate, mean_return, _success_rate_per_task, _ = evaluation(
            agent, eval_envs, evaluation_episodes
        )
        total_mean_success_rate += mean_success_rate
        total_mean_return += mean_return
        success_rate_per_task[i] = np.array(list(_success_rate_per_task.values()))

    success_rates = (success_rate_per_task).mean(axis=0)
    task_success_rates = {
        task_name: success_rates[i] for i, task_name in enumerate(set(task_names))
    }

    return (
        total_mean_success_rate / num_evals,
        total_mean_return / num_evals,
        task_success_rates,
    )


class Rollout(NamedTuple):
    observations: npt.NDArray
    actions: npt.NDArray
    rewards: npt.NDArray
    dones: npt.NDArray

    # Auxiliary policy outputs
    log_probs: npt.NDArray | None = None
    means: npt.NDArray | None = None
    stds: npt.NDArray | None = None
    values: npt.NDArray | None = None


class _MultiTaskRolloutBuffer:
    """A buffer to accumulate rollouts for multiple tasks.
    Useful for ML1, ML10, ML45, or on-policy MTRL algorithms.

    In Metaworld, all episodes are as long as the time limit (typically 500), thus in this buffer we assume
    fixed-length episodes and leverage that for optimisations."""

    rollouts: list[list[Rollout]]

    def __init__(
        self,
        num_tasks: int,
        rollouts_per_task: int,
        max_episode_steps: int,
        envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    ):
        self.num_rollout_steps = rollouts_per_task * max_episode_steps
        self.num_tasks = num_tasks
        self._obs_shape = np.array(envs.single_observation_space.shape).prod()
        self._action_shape = np.array(envs.single_action_space.shape).prod()

        self.reset()

    def reset(self) -> None:
        """Reinitialize the buffer."""
        self.observations = np.zeros(
            (self.num_rollout_steps, self.num_tasks, self._obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.num_rollout_steps, self.num_tasks, self._action_shape),
            dtype=np.float32,
        )
        self.rewards = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=np.float32
        )
        self.dones = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=np.float32
        )

        self.log_probs = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=np.float32
        )
        self.values = np.zeros_like(self.rewards)
        self.means = np.zeros_like(self.actions)
        self.stds = np.zeros_like(self.actions)
        self.pos = 0

    @property
    def ready(self) -> bool:
        """Returns whether or not a full batch of rollouts for each task has been sampled."""
        return self.pos == self.num_rollout_steps

    def get(
        self,
    ) -> Rollout:
        """Compute returns and advantages for the collected rollouts.

        Returns a Rollout tuple where each array has the batch dimensions (Timestep,Task,).
        The timesteps are multiple rollouts flattened into one time dimension."""
        rollouts = Rollout(
            self.observations,
            self.actions,
            self.rewards,
            self.dones,
            self.log_probs,
            self.means,
            self.stds,
            self.values,
        )

        return rollouts

    def add(
        self,
        obs: npt.NDArray,
        action: npt.NDArray,
        reward: npt.NDArray,
        done: npt.NDArray,
        value: npt.NDArray | None = None,
        log_prob: npt.NDArray | None = None,
        mean: npt.NDArray | None = None,
        std: npt.NDArray | None = None,
    ):
        """Add a batch of timesteps to the buffer."""
        # NOTE: assuming batch dim = task dim
        assert (
            obs.ndim == 2 and action.ndim == 2 and reward.ndim <= 2 and done.ndim <= 2
        )
        assert (
            obs.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == done.shape[0]
            == self.num_tasks
        )

        self.observations[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward.copy().reshape(-1, 1)
        self.dones[self.pos] = done.copy().reshape(-1, 1)

        if value is not None:
            self.values[self.pos] = value.copy()
        if log_prob is not None:
            self.log_probs[self.pos] = log_prob.reshape(-1, 1).copy()
        if mean is not None:
            self.means[self.pos] = mean.copy()
        if std is not None:
            self.stds[self.pos] = std.copy()

        self.pos += 1
