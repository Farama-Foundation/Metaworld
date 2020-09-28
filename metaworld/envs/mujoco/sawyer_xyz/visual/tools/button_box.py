import numpy as np

from .tool import Artifact


class ButtonBox(Artifact):
    @property
    def bbox(self):
        return np.array([-.13, -.13, -.1, .13, .13, .18])

    @property
    def resting_pos_z(self):
        return .115
