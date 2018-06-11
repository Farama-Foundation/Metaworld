import abc
from collections import OrderedDict
import numpy as np


class MultitaskEnv(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_goal(self):
        pass

    """
    Implement the batch-version of these functions.
    """
    @abc.abstractmethod
    def sample_goals(self, batch_size):
        pass

    @abc.abstractmethod
    def compute_rewards(self, achieved_goals, desired_goals, infos):
        """
        :param achieved_goals: BATCH x GOAL_DIM np array
        :param desired_goals: BATCH x GOAL_DIM np array
        :param infos: map from key to BATCH x K np arrays
        :return: np.array of shape BATCH of rewards
        """
        pass

    def sample_goal(self):
        return self.sample_goals(1)[0]

    def compute_reward(self, achieved_goal, desired_goal, info):

        if info is None:
            infos = None
        else:
            infos = {}
            for k in info.keys():
                infos[k] = np.array([info[k]])
        return self.compute_rewards(
            achieved_goal[None], desired_goal[None], infos
        )[0]

    def get_diagnostics(self, *args, **kwargs):
        """
        :param rollouts: List where each element is a dictionary describing a
        rollout. Typical dictionary might look like:
        {
            'observations': np array,
            'actions': np array,
            'next_observations': np array,
            'rewards': np array,
            'terminals': np array,
            'env_infos': list of dictionaries,
            'agent_infos': list of dictionaries,
        }
        :return: OrderedDict. Statistics to save.
        """
        return OrderedDict()
