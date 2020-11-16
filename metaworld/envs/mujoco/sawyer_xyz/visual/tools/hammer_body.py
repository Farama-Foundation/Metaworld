import numpy as np

from .tool import Tool


class HammerBody(Tool):
    @property
    def bbox(self):
        return np.array([-.2, -.05, -.05, .2, .05, .05])

    @property
    def resting_pos_z(self):
        return .06
