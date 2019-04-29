import abc

import gym
import gym.spaces

from multiworld.world import ParametricWorld

class MutableDynamics(gym.Env, metaclass=abc.ABCMeta):
    """An environment with dynamics which can change.

    Attributes:
        dynamics_space (:obj:`gym.Space'): A space representing valid dynamics
            parameters in this environment
    """
    @property
    @abc.abstractmethod
    def dynamics(self):
        """An object representing this environment's dynamics.

        Setting this property immediately changes the dynamics behavior of the
        environment.

        Notes:
            If you require users `reset()` the environment after setting this
            property, you should document this explicitly.
        """
        pass

    @dynamics.setter
    @abc.abstractmethod
    def dynamics(self, d):
        pass

    @abc.abstractmethod
    def transition(self, action):
        """Updates the environment state based on the current dynamics and the
        action.

        Args:
           action (object): An action in `self.action_space`
        """
        pass


class MutableDynamicsWorld(MutableDynamics, ParametricWorld):
    """A World whose POMDP is characterized by a set of mutable dynamics
    parameters.
    """
    def __init__(self):
        self.pomdp_space = self.dynamics_space

    @property
    def pomdp(self):
        return self.dynamics

    @pomdp.setter
    def pomdp(self, pomdp_descriptor):
        self.dynamics = pomdp_descriptor
