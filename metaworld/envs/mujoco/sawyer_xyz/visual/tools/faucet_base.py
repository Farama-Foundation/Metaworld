import numpy as np

from .tool import Tool


class FaucetBase(Tool):
    @property
    def bbox(self):
        return np.array([-.19, -.17, .1, .17, .02, .15])
