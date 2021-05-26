import numpy as np

from ._tool import Artifact


class NailBox(Artifact):
    @property
    def bbox(self):
        return np.array([-.11, -.22, .0, .11, .1, .22])
