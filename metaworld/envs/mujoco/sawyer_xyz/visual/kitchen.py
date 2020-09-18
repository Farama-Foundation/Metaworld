import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.visual.tool import Artifact, Tool


class CoffeeMug(Tool):
    @property
    def degrees_of_freedom(self):
        return [1, 1, 1, 1, 1, 1]

    @property
    def rigid_body_bbox(self):
        return np.array([-.025, -.025, 0., .05, .025, 0.07])

    @property
    def resting_pos_z(self):
        return 0.1


class CoffeeMachine(Artifact):
    @property
    def rigid_body_bbox(self):
        return np.array([-.09, -.24, .0, .09, .1, 0.37])


class Dial(Tool):
    @property
    def degrees_of_freedom(self):
        return [0, 0, 0, 0, 0, 1]

    @property
    def rigid_body_bbox(self):
        return np.array([-.04, -.04, .0, .04, .04, .1])


class FaucetBase(Tool):
    @property
    def degrees_of_freedom(self):
        return [0, 0, 0, 0, 0, 1]

    @property
    def rigid_body_bbox(self):
        return np.array([-.19, -.17, .1, .17, .02, .15])


class ToasterHandle(Tool):
    @property
    def degrees_of_freedom(self):
        return [0, 0, 1, 0, 0, 0]

    @property
    def rigid_body_bbox(self):
        return np.array([-.1, -.2, .0, .1, .1, .23])

    @property
    def resting_pos_z(self):
        return 0.37
