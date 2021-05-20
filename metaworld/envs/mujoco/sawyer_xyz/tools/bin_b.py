import numpy as np

from .tool import Artifact


class BinB(Artifact):
    @property
    def bbox(self):
        return np.array([-.1, -.1, .0, .1, .1, .12])
