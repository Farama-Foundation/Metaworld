import gym
import abc
from collections import OrderedDict


class MultitaskEnv(gym.Env, metaclass=abc.ABCMeta):
    """
    Effectively a gym.GoalEnv, but we add three more functions:

        - get_goal
        - sample_goals
        - compute_rewards

    We also change the compute_reward interface to take in an action and
    observation dictionary.
    """
    @abc.abstractmethod
    def get_goal(self):
        """
        Returns a dictionary
        """
        pass

    """
    Implement the batch-version of these functions.
    """
    @abc.abstractmethod
    def sample_goals(self, batch_size):
        """
        :param batch_size:
        :return: Returns a dictionary mapping desired goal keys to arrays of
        size BATCH_SIZE x Z, where Z depends on the key.
        """
        pass

    @abc.abstractmethod
    def compute_rewards(self, actions, obs):
        """
        :param actions: Np array of actions
        :param obs: Batch dictionary
        :return:
        """

        pass

    def sample_goal(self):
        goals = self.sample_goals(1)
        return self.unbatchify_dict(goals, 0)

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs)[0]

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

    @staticmethod
    def unbatchify_dict(batch_dict, i):
        """
        :param batch_dict: A batch dict is a dict whose values are batch.
        :return: the dictionary returns a dict whose values are just elements of
        the batch.
        """
        new_d = {}
        for k in batch_dict.keys():
            new_d[k] = batch_dict[k][i]
        return new_d

    @staticmethod
    def batchify_dict(batch_dict, i):
        """
        :param batch_dict: A batch dict is a dict whose values are batch.
        :return: the dictionary returns a dict whose values are just elements of
        the batch.
        """
        new_d = {}
        for k in batch_dict.keys():
            new_d[k] = batch_dict[k][i]
        return new_d
