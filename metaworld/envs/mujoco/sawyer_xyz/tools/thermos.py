import numpy as np

from ._tool import Tool


class Thermos(Tool):
    @property
    def bbox(self):
        return np.array([-.07, -.14, 0., .07, .05, .33])
