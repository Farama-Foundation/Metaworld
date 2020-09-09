import abc

import numpy as np


class Tool(abc.ABC):
    @property
    @abc.abstractmethod
    def degrees_of_freedom(self):
        # e.g. [1, 1, 1, 1, 1, 1] for hammer
        # e.g. [0, 0, 0, 0, 0, 1] for dial
        # e.g. [0, 0, 0, 0, 0, 0] for the box in peg-insert
        # Note that we always ignore roll and pitch, all current tools
        # only care about yaw
        pass

    @property
    @abc.abstractmethod
    def rigid_body_bbox_size(self):
        pass


class SawyerGripper(Tool):
    @property
    def degrees_of_freedom(self):
        return 1, 1, 1, 0, 0, 0

    @property
    def rigid_body_bbox_size(self):
        return np.array((0.05, 0.07, .14))


class Artifact(Tool, abc.ABC):
    @property
    def degrees_of_freedom(self):
        return [0, 0, 0, 0, 0, 0]
