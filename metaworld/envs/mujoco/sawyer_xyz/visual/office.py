import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.visual.tool import Artifact, Tool


class BinLid(Tool):
    @property
    def degrees_of_freedom(self):
        return [1, 1, 1, 1, 1, 1]

    @property
    def rigid_body_bbox(self):
        return np.array([-.12, -.12, .06, .12, .12, .16])


class BinA(Artifact):
    @property
    def rigid_body_bbox(self):
        return np.array([-.1, -.1, .0, .1, .1, .06])


class BinB(Artifact):
    @property
    def rigid_body_bbox(self):
        return np.array([-.1, -.1, .0, .1, .1, .06])


class Door(Artifact):
    @property
    def degrees_of_freedom(self):
        return [0, 0, 0, 0, 0, 1]

    @property
    def rigid_body_bbox(self):
        return np.array([-.22, -.55, -.15, .22, .10, .15])


class Drawer(Artifact):
    @property
    def degrees_of_freedom(self):
        return [0, 1, 0, 0, 0, 0]

    @property
    def rigid_body_bbox(self):
        return np.array([-.12, -.33, .0, .12, .1, .17])


class Shelf(Artifact):
    @property
    def rigid_body_bbox(self):
        return np.array([-.12, -.1, 0., .12, .1, .65])


class Window(Artifact):
    @property
    def degrees_of_freedom(self):
        return [1, 0, 0, 0, 0, 0]

    @property
    def rigid_body_bbox(self):
        return np.array([-.19, -.1, .0, .19, .05, .49])
