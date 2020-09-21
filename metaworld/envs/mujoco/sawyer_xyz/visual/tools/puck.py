import numpy as np

from .tool import Tool


class Puck(Tool):
    @property
    def bbox(self):
        return np.array([-.015, -.015, .0, .015, .015, .05])

    @property
    def resting_pos_z(self):
        return .04
