import abc
from collections import OrderedDict


class MultitaskEnv(object, metaclass=abc.ABCMeta):
    """
    An environment with multiple goals.

    The interface is much like a gym.Env:

    ```
    obs = env.reset()  # internally sets a new goal
    next_observation, reward, done, info = env.step(action)
    goal = env.get_goal()
    ```

    We make liberal use of the "info" dictionary.
    In particular, it should contain the goal.
    """

    @abc.abstractmethod
    def get_goal(self):
        pass

    def get_info(self):
        """
        Ideally we'd change reset to return an observation and an info dict,
        but this would break the gym interface.

        Example use case:

        ```
        obs = env.reset()
        info = env.get_info()

        ```
        """
        return {}

    """
    Implement the batch-version of these functions.
    """
    @abc.abstractmethod
    def sample_goals(self, batch_size):
        pass

    @abc.abstractmethod
    def compute_rewards(self, obs, actions, next_obs, goals, env_infos):
        pass

    def sample_goal(self):
        return self.sample_goals(1)[0]

    def compute_reward(self, ob, action, next_ob, goal, env_info):
        return self.compute_rewards(
            ob[None], action[None], next_ob[None], goal[None], [env_info],
        )

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
