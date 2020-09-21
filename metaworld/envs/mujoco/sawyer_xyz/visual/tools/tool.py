import abc
import itertools


def get_position_of(tool, mjsim):
    return mjsim.data.get_body_xpos(tool.name)


def set_position_of(tool, mjsim, mjmodel):
    assert(len(tool.specified_pos) == 3)
    mjsim.model.body_pos[mjmodel.body_name2id(tool.name)] = tool.specified_pos


class Tool(abc.ABC):
    def __init__(self):
        self.specified_pos = None

    @property
    def name(self):
        return type(self).__name__

    @property
    def pos_is_static(self):
        return False

    @property
    @abc.abstractmethod
    def bbox(self):
        pass

    def get_bbox_corners(self):
        bbox = self.bbox.tolist()
        assert len(bbox) == 6
        return itertools.product(*zip(bbox[:3], bbox[3:]))

    @property
    def resting_pos_z(self):
        return 0.0


class Artifact(Tool, abc.ABC):
    @property
    def pos_is_static(self):
        return True
