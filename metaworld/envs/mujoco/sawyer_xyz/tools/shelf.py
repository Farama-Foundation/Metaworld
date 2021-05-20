import numpy as np

from .tool import Artifact


class Shelf(Artifact):
    @property
    def bbox(self):
        return np.array([-.12, -.1, 0., .12, .1, .65])
