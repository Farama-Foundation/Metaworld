import abc
import itertools

import numpy as np


def get_position_of(tool, mjsim):
    return mjsim.data.get_body_xpos(tool.name)


def set_position_of(tool, pos, mjsim, mjmodel):
    assert(len(pos) == 3)
    mjsim.model.body_pos[mjmodel.body_name2id(tool.name)] = pos


class Tool(abc.ABC):
    @property
    def name(self):
        return type(self).__name__

    @property
    def degrees_of_freedom(self):
        # e.g. [1, 1, 1, 1, 1, 1] for hammer
        # e.g. [0, 0, 0, 0, 0, 1] for dial
        # e.g. [0, 0, 0, 0, 0, 0] for the box in peg-insert
        # Note that we always ignore roll and pitch, all current tools
        # only care about yaw
        return [1, 1, 1, 1, 1, 1]

    @property
    @abc.abstractmethod
    def rigid_body_bbox(self):
        pass

    def get_bbox_corners(self):
        bbox = self.rigid_body_bbox.tolist()
        assert len(bbox) == 6
        return itertools.product(*zip(bbox[:3], bbox[3:]))

    @property
    def resting_pos_z(self):
        return 0.0


class SawyerGripper(Tool):
    @property
    def name(self):
        return ''

    @property
    def degrees_of_freedom(self):
        return 1, 1, 1, 0, 0, 0

    @property
    def rigid_body_bbox(self):
        # TODO get actual numbers
        return np.array([0.0, 0.0, 0.0, 0.05, 0.07, .14])


class Artifact(Tool, abc.ABC):
    @property
    def degrees_of_freedom(self):
        return [0, 0, 0, 0, 0, 0]
