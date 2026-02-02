from abc import ABC, abstractmethod

import numpy as np

from metaworld.policies import ENV_POLICY_MAP

import gymnasium as gym


class MetaworldAgent(ABC):
    @abstractmethod
    def get_action(self, obs: np.ndarray, info: dict, task_name: str, action_space) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self):
        pass


class RandomMetaworldAgent(MetaworldAgent):
    def __init__(self, seed: int = None):
        if seed is None:
            self.seed = 42
        self.seed = seed
        self.reset()

    def get_action(self, obs: np.ndarray, info: dict, task_name: str, action_space) -> np.ndarray:
        low = action_space.low
        high = action_space.high
        return self.rng.uniform(low, high)

    def reset(self):
        self.rng = np.random.default_rng(self.seed)


class ExpertPolicyMetaworldAgent(MetaworldAgent):
    def get_action(self, obs, info, task_name: str, action_space):
        if task_name is None:
            raise ValueError(
                "Task name must be provided for ExpertPolicyMetaworldAgent.")
        if self.policy_task_name != task_name:
            self.policy_task_name = task_name
            policy_cls = ENV_POLICY_MAP[task_name]
            self.policy = policy_cls()
        return self.policy.get_action(obs)

    def reset(self):
        self.policy_task_name = None
        self.policy = None


def run_agent_episode_in_env(env: gym.Env,
                             agent: MetaworldAgent,
                             max_episode_steps: int,
                             record_keys: set[str] | None = None,) -> dict:

    if record_keys is None:
        record_keys = set()

    # If we run N steps, we record N+1 observations (initial + N results).
    buffer_size = max_episode_steps + 1

    rec_obs = 'observations' in record_keys
    rec_rewards = 'rewards' in record_keys
    rec_terminates = 'terminates' in record_keys
    rec_truncates = 'truncates' in record_keys
    rec_agent_actions = 'agent_actions' in record_keys

    agent.reset()
    obs, reset_info = env.reset()
    info = reset_info

    # Preallocate arrays
    # Observations
    if rec_obs:
        observations = np.zeros(
            (buffer_size, *obs.shape), dtype=obs.dtype)

    # Actions (Note: buffer_size is enough, though actions will be 1 less than obs)
    if rec_agent_actions:
        agent_actions = np.zeros(
            (buffer_size, *env.action_space.shape), dtype=env.action_space.dtype)

    # Scalars (floats initialized to NaN, bools to False)
    if rec_rewards:
        rewards = np.full(buffer_size, np.nan, dtype=np.float64)
    if rec_terminates:
        terminates = np.zeros(buffer_size, dtype=bool)
    if rec_truncates:
        truncates = np.zeros(buffer_size, dtype=bool)

    agent_first_success_step = None
    any_terminated = False
    any_truncated = False

    done = False
    agent_step = 0
    while True:
        # Record Pre-Step Data
        if rec_obs:
            observations[agent_step] = obs

        if done:
            break

        agent_action = agent.get_action(
            obs, info, env.unwrapped.ENV_NAME, env.action_space)

        if rec_agent_actions:
            agent_actions[agent_step] = agent_action

        obs, reward, terminate, truncate, info = env.step(
            agent_action)
        step_is_success = info.get('success', 0.0) >= 1.0

        if terminate:
            any_terminated = True

        if truncate:
            any_truncated = True

        if step_is_success and agent_first_success_step is None:
            agent_first_success_step = agent_step

        done = terminate or truncate

        # Record Post-Step Data
        if rec_rewards:
            rewards[agent_step] = reward
        if rec_terminates:
            terminates[agent_step] = terminate
        if rec_truncates:
            truncates[agent_step] = truncate

        agent_step += 1

    env.close()

    # Determine slice indices
    obs_slice = agent_step + 1
    trans_slice = agent_step

    ret = {}
    if rec_obs:
        ret['observations'] = observations[:obs_slice]
    if rec_rewards:
        ret['rewards'] = rewards[:trans_slice]
    if rec_terminates:
        ret['terminates'] = terminates[:trans_slice]
    if rec_truncates:
        ret['truncates'] = truncates[:trans_slice]

    if rec_agent_actions:
        ret['agent_actions'] = agent_actions[:trans_slice]

    ret['env_name'] = env.unwrapped.ENV_NAME
    ret['env_seed'] = reset_info['seed']
    ret['agent_first_success_step'] = agent_first_success_step
    ret['total_episode_steps'] = agent_step
    ret['any_terminated'] = any_terminated
    ret['any_truncated'] = any_truncated

    return ret


def run_agent_episode(env_name,
                      seed,
                      agent: MetaworldAgent,
                      max_episode_steps: int,
                      record_keys: set[str] | None = None,
                      reward_function_version: str = 'v2',
                      ) -> dict:
    env = gym.make('Meta-World/MT1',
                   env_name=env_name,
                   seed=seed,
                   reward_function_version=reward_function_version,
                   max_episode_steps=max_episode_steps,
                   num_tasks_per_env=1,
                   )

    return run_agent_episode_in_env(
        env=env,
        agent=agent,
        max_episode_steps=max_episode_steps,
        record_keys=record_keys,
    )
