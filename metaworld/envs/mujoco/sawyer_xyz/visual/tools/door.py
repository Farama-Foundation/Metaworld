import numpy as np

from .tool import Artifact


class Door(Artifact):
    @property
    def bbox(self):
        return np.array([-.22, -.55, -.15, .22, .10, .15])

    @property
    def resting_pos_z(self):
        return .15
