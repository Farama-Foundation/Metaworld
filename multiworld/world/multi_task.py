import abc

import gym
import gym.spaces

from multiworld.world import ParametricWorld


class MultiTask(gym.Env, metaclass=abc.ABCMeta):
    """A multi-task environment.

    Attributes:
        task_space (:obj:`gym.Space`): A space representing valid tasks in this
            environment.
    """
    @property
    @abc.abstractmethod
    def task(self):
        pass

    @task.setter
    @abc.abstractmethod
    def task(self, t):
        pass

    @abc.abstractmethod
    def reward(self, state, action, next_state):
        """Computes the reward for the MDP transition (s, a, s') under the
        current task.

        Returns:
            float: Reward under the current task for the provided transition
        """
        pass


class DiscreteMultiTask(MultiTask, metaclass=abc.ABCMeta):
    """A multi-task environment, where each task is a reward function chosen
    from a discrete set.

    Reward functions should be static method which implement the prototype:
        @staticmethod
        def reward_fn(state, action, next_state):
            '''A reward function.

            Args:
                state (object): MDP state before the transition
                action (object): Action causing the transition
                next_state (object): MDP state after the transition

            Return:
                float: Reward for this transition
            '''
            pass
    """
    def __init__(self, *reward_functions):
        self._reward_functions = reward_functions
        self._task = 0
        self.task_space = gym.spaces.Discrete(len(self._reward_functions))

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, t):
        self._task = t

    def reward(self, state, action, next_state):
        return self._reward_functions[self._task](state, action, next_state)



class MultiTaskWorld(MultiTask, ParametricWorld):
    """A World whose POMDP is characterized by a space of reward functions."""

    def __init__(self):
        ParametricWorld.__init__(self)
        self.pomdp_space = self.task_space

    @property
    def pomdp(self):
        return self.task

    @pomdp.setter
    def pomdp(self, pomdp_descriptor):
        self.task = pomdp_descriptor