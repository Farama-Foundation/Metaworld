import numpy as np

from .tool import Artifact


class NailBox(Artifact):
    @property
    def bbox(self):
        return np.array([-.1, -.22, .0, .1, .1, .22])
