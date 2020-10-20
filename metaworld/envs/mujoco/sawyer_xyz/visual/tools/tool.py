import abc
import itertools


def get_position_of(tool, mjsim):
    return mjsim.data.get_body_xpos(tool.name)


def set_position_of(tool, mjsim, mjmodel):
    assert(len(tool.specified_pos) == 3)
    mjsim.model.body_pos[mjmodel.body_name2id(tool.name)] = tool.specified_pos


def get_joint_pos_of(tool, mjsim):
    return mjsim.data.get_joint_qpos(tool.name + 'Joint')


def set_joint_pos_of(tool, mjsim, pos):
    mjsim.data.set_joint_qpos(tool.name + 'Joint', pos)


def get_joint_vel_of(tool, mjsim):
    return mjsim.data.get_joint_qvel(tool.name + 'Joint')


def set_joint_vel_of(tool, mjsim, vel):
    mjsim.data.set_joint_qvel(tool.name + 'Joint', vel)


def get_quat_of(tool, mjsim):
    return mjsim.data.get_body_xquat[tool.name]


def set_quat_of(tool, mjsim, mjmodel):
    assert(len(tool.specified_quat) == 4)
    mjsim.model.body_quat[mjmodel.body_name2id(tool.name)] = tool.specified_quat
    return


def get_vel_of(tool, mjsim, mjmodel):
    return mjsim.data.cvel[mjmodel.body_name2id(tool.name)]


def set_vel_of(tool, mjsim, mjmodel):
    '''
    mjmodel does not have access to set cvel values since they are calculated:
    mjsim.model.cvel[mjmodel.body_name2id(tool.name)] = new_vel
    '''
    raise NotImplementedError


class Tool(abc.ABC):
    def __init__(self):
        self.specified_pos = None
        self.specified_quat = None

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
