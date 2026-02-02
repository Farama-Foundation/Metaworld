from abc import ABC, abstractmethod

import numpy as np


class MetaworldAgent(ABC):
    @abstractmethod
    def get_action(self, obs: np.ndarray, info: dict, action_space) -> np.ndarray:
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

    def get_action(self, obs: np.ndarray, info: dict, action_space) -> np.ndarray:
        low = action_space.low
        high = action_space.high
        return self.rng.uniform(low, high)

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
