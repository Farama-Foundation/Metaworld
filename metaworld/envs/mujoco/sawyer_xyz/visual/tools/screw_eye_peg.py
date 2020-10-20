import numpy as np

from .tool import Artifact


class ScrewEyePeg(Artifact):
    @property
    def bbox(self):
        return np.array([-.01, -.01, -.05, .01, .01, .05])

    @property
    def resting_pos_z(self):
        return .05
