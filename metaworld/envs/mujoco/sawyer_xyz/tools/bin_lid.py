import numpy as np

from .tool import Tool


class BinLid(Tool):
    @property
    def bbox(self):
        return np.array([-.12, -.12, .0, .12, .12, .2])

    @property
    def resting_pos_z(self):
        return .1
