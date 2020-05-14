import gym
import abc


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

    @abc.abstractmethod
    def sample_goal(self):
        pass

    @abc.abstractmethod
    def compute_reward(self, action, obs):
        pass
