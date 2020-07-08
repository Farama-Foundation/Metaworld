import abc
import warnings

import numpy as np


def assert_fully_parsed(func):
    """Decorator function to ensure observations are fully parsed

    Args:
        func (Callable): The function to check

    Returns:
        (Callable): The input function, decorated to assert full parsing
    """
    def inner(obs):
        obs_dict = func(obs)
        assert len(obs) == sum(
            [len(i) if isinstance(i, np.ndarray) else 1 for i in obs_dict.values()]
        ), 'Observation not fully parsed'
        return obs_dict
    return inner


def move(from_xyz, to_xyz, p):
    """Computes action components that help move from 1 position to another

    Args:
        from_xyz (np.ndarray): The coordinates to move from (usually current position)
        to_xyz (np.ndarray): The coordinates to move to
        p (float): constant to scale response

    Returns:
        (np.ndarray): Response that will decrease abs(to_xyz - from_xyz)

    """
    error = to_xyz - from_xyz
    response = p * error

    if np.any(np.absolute(response) > 1.):
        warnings.warn('Constant(s) may be too high. Environments clip response to [-1, 1]')

    return response


class Policy(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def _parse_obs(obs):
        """Pulls pertinent information out of observation and places in a dict.

        Args:
            obs (np.ndarray): Observation which conforms to env.observation_space

        Returns:
            dict: Dictionary which contains information from the observation
        """
        pass

    @abc.abstractmethod
    def get_action(self, obs):
        """Gets an action in response to an observation.

        Args:
            obs (np.ndarray): Observation which conforms to env.observation_space

        Returns:
            np.ndarray: Array (usually 4 elements) representing the action to take
        """
        pass
