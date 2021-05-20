import numpy as np

from .tool import Artifact


class BasketballHoop(Artifact):
    @property
    def bbox(self):
        return np.array([-.1, -.14, 0., .1, .01, .4])
