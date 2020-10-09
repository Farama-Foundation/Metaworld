import numpy as np

from .tool import Tool


class ElectricalPlug(Tool):
    @property
    def bbox(self):
        return np.array([.0, -.02, -.02, .1, .02, .02])

    @property
    def resting_pos_z(self):
        return .03
