import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.visual.tool import Artifact, Tool


class ButtonBox(Artifact):
    @property
    def rigid_body_bbox(self):
        return np.array([-.13, -.13, -.1, .13, .13, .18])


class ElectricalPlug(Tool):
    @property
    def rigid_body_bbox(self):
        return np.array([.0, -.02, -.02, .1, .02, .02])

    @property
    def resting_pos_z(self):
        return 0.351


class ElectricalOutlet(Artifact):
    @property
    def rigid_body_bbox(self):
        return np.array([.0, -.09, .0, .24, .09, .22])

    @property
    def resting_pos_z(self):
        return 0.22


class HammerBody(Tool):
    @property
    def rigid_body_bbox(self):
        return np.array([.0, .0, .0, .4, .1, .1])

    @property
    def resting_pos_z(self):
        return 0.02


class NailBox(Artifact):
    @property
    def rigid_body_bbox(self):
        return np.array([-.1, -.22, .0, .1, .1, .22])


class Lever(Artifact):
    @property
    def degrees_of_freedom(self):
        return [0, 0, 0, 0, 1, 0]

    @property
    def rigid_body_bbox(self):
        return np.array([-.05, -.22, .0, .14, .05, .6])


class ScrewEye(Tool):
    @property
    def rigid_body_bbox(self):
        return np.array([-.06, -.05, -.02, .17, .05, .02])

    @property
    def resting_pos_z(self):
        return 0.02


class ScrewEyePeg(Artifact):
    @property
    def rigid_body_bbox(self):
        return np.array([-.01, -.01, -.05, .01, .01, .05])

    @property
    def resting_pos_z(self):
        return 0.05
