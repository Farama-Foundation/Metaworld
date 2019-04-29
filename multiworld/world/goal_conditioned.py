import abc

import gym

from multiworld.world import ParametricWorld


class GoalConditioned(gym.Env, metaclass=abc.ABCMeta):
    """A goal-conditioned environment.

    Attributes:
        goal_space (:obj:`gym.Space`): A space representing valid goals states
            in this environment.
    """
    @property
    @abc.abstractmethod
    def goal(self):
        """The goal state with which this environment calculates rewards.

        Setting this property immediately changes the rewards calculated by the
        environment.

        Notes:
            If you require users `reset()` the environment after setting this
            property, you should document this explicitly.
        """
        pass

    @goal.setter
    @abc.abstractmethod
    def goal(self, value):
        pass


class GoalConditionedWorld(GoalConditioned, ParametricWorld):
    """A World whose POMDP is characterized by a goal state."""

    def __init__(self):
        self.pomdp_space = self.goal_space

    @property
    def pomdp(self):
        return self.goal

    @pomdp.setter
    def pomdp(self, pomdp_descriptor):
        self.goal = pomdp_descriptor