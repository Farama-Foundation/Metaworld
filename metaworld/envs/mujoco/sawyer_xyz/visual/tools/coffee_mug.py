import numpy as np

from .tool import Tool


class CoffeeMug(Tool):
    @property
    def bbox(self):
        return np.array([-.025, -.025, 0., .05, .025, .07])

    @property
    def resting_pos_z(self):
        return .05
