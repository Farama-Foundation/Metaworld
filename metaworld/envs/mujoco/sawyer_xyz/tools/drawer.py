import numpy as np

from ._tool import Artifact


class Drawer(Artifact):
    @property
    def bbox(self):
        return np.array([-.12, -.33, .0, .12, .1, .17])
