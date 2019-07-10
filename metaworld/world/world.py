import abc
import pprint

import gym


class POMDPDescriptor(abc.ABC):
    """A representation of the family of POMDPs which a :obj:`World` can
    execute.

    This class primarily exists to document the `POMDPDescriptor` interface,
    and inherting your descriptors from it is optional, as long as your
    descriptor implements the interface described here.

    Required properties:
        * **Comparable**. Given :obj:`POMDPDescriptor` instances `a` and `b`
          representing the same POMDP, it must be that `a == b`. Likewise, if
          `a` and `b` represent different POMDPs, it must be that `a != b`.
            >>> from some.where import SamePOMDP
            >>> from some.where import DifferentPOMDP
            >>> a = SamePOMDP()
            >>> b = SamePOMDP()
            >>> c = DifferentPOMDP()
            >>> assert a == b
            >>> assert a != c
            >>> assert b != c
        * **Pickleable**, i.e.
            >>> import pickle
            >>> from some.where import CorrectDescriptor
            >>> c = CorrectDescriptor()
            >>> assert c == pickle.loads(pickle.dumps(c))

    Recommended properties:
        * **Serializable** to a text format wherever possible, e.g.
            >>> import json
            >>> from some.where import BestDescriptor
            >>> d = BestDescriptor()
            >>> assert d == json.loads(json.dumps(d))
        * **Immutable**. This is difficult to implement in Python without
          making the interface cumbersome to use, but a :obj:`World` should
          treat a :obj:`POMDPDescriptor` as though it is immutable, and may
          exhibit undefined behavior if the :obj:`POMDPDescriptor` is mutated.
    """
    pass


class BaseWorld(gym.Env, metaclass=abc.ABCMeta):
    """Represents a family of (PO)MDPs implementing the :obj:`gym.Env`
    interface.

    A :obj:`World` executes a *single* POMDP at any given time.

    The family of POMDPs a :obj:`World` represents is encoded by a POMDP
    descriptor. The descriptor encodes all variations of the POMDPs in this
    :obj:`World`. The descriptor may be any pickleable object.

    No API other than :obj:`World.pomdp` should change the current POMDP. That
    is, a :obj:`World` should not mutate its POMDP as a result of calling
    :obj:`step()`, :obj:`reset()`, or any other API. Following from above,
    :obj:`__init__()` should not have parameters which change the POMDP
    behavior.

    It is preferred that calling :obj:`reset()` is not required after setting
    `pomdp` (i.e. changing the POMDP does not change the hidden state), but if
    a :obj:`World` imposes this requirement, it should be clearly documented.
    """
    @property
    @abc.abstractmethod
    def pomdp(self):
        """The descriptor representing the POMDP currently being executed by
        this :obj:`World`.

        Setting this property changes the current descriptor, and therefore the
        :obj:`World`'s POMDP behavior, immediately.
        """
        pass

    @pomdp.setter
    @abc.abstractmethod
    def pomdp(self, pomdp_descriptor):
        pass


class World(BaseWorld, metaclass=abc.ABCMeta):

    @classmethod
    def from_pomdp(cls, pomdp_descriptor):
        """Construct a :obj:`World` from a :obj:`POMDPDescriptor`.

        Args:
            pomdp_descriptor (:obj:`POMDPDescriptor`): An object representing a
            POMDP in this :obj:`World`

        Returns:
            A new instance of this :obj:`World` which executes the POMDP
            represented by `pomdp_descriptor`
        """
        world = cls()
        world.pomdp = descriptor
        return world

    @property
    def pomdp(self):
        return self._pomdp

    @pomdp.setter
    def pomdp(self, pomdp_descriptor):
        self._pomdp = pomdp_descriptor

    def __getstate__(self):
        data = super().__getstate__()
        data['pomdp'] = self.pomdp

    def __setstate__(self, data):
        super().__setstate__(data)
        self.pomdp = data['pomdp']

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.pomdp)


class ParametricWorld(World, metaclass=abc.ABCMeta):
    """A :obj:`World` whose POMDPs can be described by a :obj:`gym.Space`
    object.

    Attributes:
        pomdp_space (:obj:`gym.Space`): A space which describes valid POMDP
            descriptors for this :obj:`ParametricWorld`.
    """

    @property
    def pomdp(self):
        return self._pomdp

    @pomdp.setter
    def pomdp(self, pomdp_descriptor):
        self._validate_descriptor(pomdp_descriptor)
        self._pomdp = pomdp_descriptor

    def _validate_descriptor(self, pomdp_descriptor):
        if not self.pomdp_space.contains(pomdp_descriptor):
            raise ValueError('descriptor must be in the space:\n'
                             '\t{}\n'
                             '\t...but you provided: \n'
                             '\t{}'.format(self.pomdp_space, pomdp_descriptor))
