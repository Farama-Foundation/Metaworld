import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.visual.tool import Artifact, Tool


class Basketball(Tool):
    @property
    def degrees_of_freedom(self):
        return [1, 1, 1, 1, 1, 1]

    @property
    def rigid_body_bbox(self):
        return np.array([-.025, -.025, -.025, .025, .025, .025])

    @property
    def resting_pos_z(self):
        return 0.03


class BasketballHoop(Artifact):
    @property
    def rigid_body_bbox(self):
        return np.array([-.1, -.14, 0., .1, .01, .4])


class Puck(Tool):
    @property
    def degrees_of_freedom(self):
        return [1, 1, 1, 1, 1, 1]

    @property
    def rigid_body_bbox(self):
        return np.array([-.015, -.015, .0, .015, .015, .05])

    @property
    def resting_pos_z(self):
        return 0.04


class SoccerGoal(Artifact):
    @property
    def rigid_body_bbox(self):
        return np.array([-.1, -.06, .0, .1, .05, .16])


class Thermos(Tool):
    @property
    def degrees_of_freedom(self):
        return [1, 1, 0, 0, 0, 1]

    @property
    def rigid_body_bbox(self):
        return np.array([-.07, -.14, 0., .07, .05, .33])
