import numpy as np

from .tool import Artifact


class Lever(Artifact):
    @property
    def bbox(self):
        return np.array([-.05, -.22, .0, .14, .05, .6])
