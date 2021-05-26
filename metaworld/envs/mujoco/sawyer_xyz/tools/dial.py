import numpy as np

from ._tool import Tool


class Dial(Tool):
    @property
    def bbox(self):
        return np.array([-.04, -.04, .0, .04, .04, .1])
