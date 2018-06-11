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
    def compute_rewards(self, obs, actions, next_obs, env_infos=None):
        """
        It's important that this function actually computes the goals!
        It shouldn't (e.g.) look up the reward in env_infos, because it might be
        stale if the goal got relabelled.

        :param obs: BATCH_SIZE x OBS_DIM numpy array
        :param actions: BATCH_SIZE x ACTION_DIM numpy array
        :param next_obs: BATCH_SIZE x obs numpy array
        :param env_infos: dictionary from string to list of length BATCH_SIZE
        :return:
        """
        pass

    def sample_goal(self):
        return self.sample_goals(1)[0]

    def compute_reward(self, ob, action, next_ob, env_info=None):
        if env_info is None:
            env_info_batch = None
        else:
            env_info_batch = {}
            for k in env_info.keys():
                env_info_batch[k] = np.array([env_info[k]])
        ob_batch = {}
        for k in ob.keys():
            ob_batch[k] = np.array([ob[k]])
        next_ob_batch = {}
        for k in next_ob.keys():
            next_ob_batch[k] = np.array([next_ob[k]])
        return self.compute_rewards(
            ob_batch, action[None], next_ob_batch, env_info_batch,
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
