from abc import ABC, abstractmethod

import numpy as np

from metaworld.policies import ENV_POLICY_MAP


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
