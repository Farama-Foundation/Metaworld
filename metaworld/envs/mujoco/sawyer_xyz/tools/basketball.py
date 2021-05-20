import numpy as np

from .tool import Tool


class Basketball(Tool):
    @property
    def bbox(self):
        return np.array([-.05, -.05, -.05, .05, .05, .05])

    @property
    def resting_pos_z(self):
        return .03
