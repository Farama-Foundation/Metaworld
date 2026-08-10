from __future__ import annotations

import abc
from typing import Any, Callable

import numpy as np
import numpy.typing as npt


def assert_fully_parsed(
    func: Callable[[npt.NDArray[np.float64]], dict[str, npt.NDArray[np.float64]]]
) -> Callable[[npt.NDArray[np.float64]], dict[str, npt.NDArray[np.float64]]]:
    """Decorator function to ensure observations are fully parsed.

    Args:
        func: The function to check

    Returns:
        The input function, decorated to assert full parsing
    """

    def inner(obs) -> dict[str, Any]:
        obs_dict = func(obs)
        assert len(obs) == sum(
            [len(i) if isinstance(i, np.ndarray) else 1 for i in obs_dict.values()]
        ), "Observation not fully parsed"
        return obs_dict

    return inner


def move(
    from_xyz: npt.NDArray[Any], to_xyz: npt.NDArray[Any], p: float
) -> npt.NDArray[Any]:
    """Computes action components that help move from 1 position to another.

    Args:
        from_xyz: The coordinates to move from (usually current position)
        to_xyz: The coordinates to move to
        p: constant to scale response

    Returns:
        Clipped response that will decrease abs(to_xyz - from_xyz)
    """
    error = to_xyz - from_xyz
    response = p * error

    return np.clip(response, -1.0, 1.0)


class Policy(abc.ABC):
    """Abstract base class for policies."""

    @staticmethod
    @abc.abstractmethod
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        """Pulls pertinent information out of observation and places in a dict.

        Args:
            obs: Observation which conforms to env.observation_space

        Returns:
            dict: Dictionary which contains information from the observation
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        """Gets an action in response to an observation.

        Args:
            obs: Observation which conforms to env.observation_space

        Returns:
            Array (usually 4 elements) representing the action to take
        """
        raise NotImplementedError
