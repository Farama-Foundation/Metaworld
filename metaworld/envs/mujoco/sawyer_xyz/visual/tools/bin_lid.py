import numpy as np

from .tool import Tool


class BinLid(Tool):
    @property
    def bbox(self):
        return np.array([-.12, -.12, .06, .12, .12, .16])
