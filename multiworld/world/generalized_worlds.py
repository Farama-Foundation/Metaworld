import gym

from multiworld.world import ParametricWorld


class MultiObservationWorld(ParametricWorld):
    """A World whose POMDP is characterized by a discrete set of observation
    functions."""

    def __init__(self, *args):
        self._observation_functions = args
        self.pomdp_space.spaces['observation_function'] = gym.spaces.Discrete(
            len(self._observation_functions))

    @property
    def _observation(self):
        return self._observation_functions[self._pomdp['observation_function']]


class MultiTransitionWorld(ParametricWorld):
    """A World whose POMDP is characterized by a discrete set of transition
    functions."""

    def __init__(self, *args):
        self._transition_functions = args
        self.pomdp_space.spaces['transition_function'] = gym.spaces.Discrete(
            len(self._transition_functions))

    @property
    def _transition(self):
        return self._transition_functions[self._pomdp['transition_function']]


class MultiRewardWorld(ParametricWorld):
    """A World whose POMDP is characterized by a discrete set of reward
    functions."""

    def __init__(self, *args):
        self._reward_functions = args
        self.pomdp_space.spaces['reward_function'] = gym.spaces.Discrete(
            len(self._reward_functions))

    @property
    def _reward(self):
        return self._reward_functions[self._pomdp['reward_function']]


class MultiTerminationWorld(ParametricWorld):
    """A World whose POMDP is characterized by a discrete set of termination
    functions."""

    def __init__(self, *args):
        self._termination_functions = args
        self.pomdp_space.spaces['termination_function'] = gym.spaces.Discrete(
            len(self._termination_functions))

    @property
    def _termination(self):
        return self._termination_functions[self._pomdp['termination_function']]


class GeneralizedWorld(MultiObservationWorld, MultiTransitionWorld,
                       MultiRewardWorld, MultiTerminationWorld):
    """A World whose POMDP is characterized by discrete sets of observation,
    transition, reward, and termination functions.

    Args:
        observation_functions (``Iterable[Callable]``): An iterable of zero or
            more observation functions.
        transition_functions (``Iterable[Callable]``): An iterable of zero or
            more transition functions.
        reward_functions (``Iterable[Callable]``): An iterable of zero or more
            reward functions.
        termination_functions (``Iterable[Callable]``): An iterable of zero or
            more termination functions.
    """

    def __init__(self,
                 observation_functions=None,
                 transition_functions=None,
                 reward_functions=None,
                 termination_functions=None):

        self.pomdp_space = gym.spaces.Dict()

        if observation_functions:
            MultiObservationWorld.__init__(self, *observation_functions)

        if transition_functions:
            MultiTransitionWorld.__init__(self, *transition_functions)

        if reward_functions:
            MultiRewardWorld.__init__(self, *reward_functions)

        if termination_functions:
            MultiTerminationWorld.__init__(self, *termination_functions)
